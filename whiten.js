// 美白 + 瘦脸。美白只按颜色找皮肤提亮；瘦脸用 MediaPipe 人脸检测（文件在本站 mp/，打开瘦脸时才加载）
// 找到脸后把两颊、下巴往里收一点点，背景不动。
// 用法：Whiten.start(video, canvas); Whiten.setLevel(0..3); await Whiten.loadFace(); Whiten.setSlim(true); Whiten.stop()
(function () {
  const LEVELS = { 1: .2, 2: .35, 3: .5 };
  let gl = null, cv = null, video = null, prog = null, uK = null, tex = null;
  let running = false, level = 2, vfcId = 0, rafId = 0;
  let uFace = null, uSlim = null, slimOn = false, fd = null, faceLoading = null, face = null, frameN = 0, lastTs = 0;
  const SLIM = .045;   // "a little": cheeks move in about 4–5%
  const here = p => new URL(p, document.baseURI).href;

  function loadFace() {
    if (faceLoading) return faceLoading;
    faceLoading = (async () => {
      const mp = await import(here('mp/vision_bundle.mjs'));
      const files = await mp.FilesetResolver.forVisionTasks(here('mp/wasm'));
      let err;
      for (const delegate of ['GPU', 'CPU']) {
        try {
          fd = await mp.FaceDetector.createFromOptions(files, {
            baseOptions: { modelAssetPath: here('mp/blaze_face_short_range.tflite'), delegate },
            runningMode: 'VIDEO', minDetectionConfidence: .5,
          });
          return;
        } catch (e) { err = e; }
      }
      throw err;
    })();
    faceLoading.catch(() => { faceLoading = null; });
    return faceLoading;
  }

  // find the face (every 3rd frame) and glide the box so it doesn't jitter.
  // The video goes in as is: a shrunken non-square copy came back with the box in the wrong place.
  function findFace() {
    const sw = video.videoWidth, sh = video.videoHeight;
    const ts = Math.max(lastTs + 1, Math.round(performance.now())); lastTs = ts;
    try {
      const d = fd.detectForVideo(video, ts).detections || [];
      let best = null;
      for (const x of d) if (!best || x.boundingBox.width > best.boundingBox.width) best = x;
      if (best) {
        const bb = best.boundingBox;
        const nf = [(bb.originX + bb.width / 2) / sw, (bb.originY + bb.height / 2) / sh, bb.width / 2 / sw, bb.height / 2 / sh];
        face = face ? face.map((v, i) => v + (nf[i] - v) * .4) : nf;
      } else if (face) {
        face[2] *= .8; face[3] *= .8; if (face[2] < .01) face = null;
      }
    } catch (e) {}
  }

  const VS = `attribute vec2 p; varying vec2 uv;
void main() { uv = vec2(p.x * .5 + .5, .5 - p.y * .5); gl_Position = vec4(p, 0., 1.); }`;
  const FS = `precision mediump float;
varying vec2 uv; uniform sampler2D vid; uniform float k, slim;
uniform vec4 face;   // centre x, y (origin top-left), half width, half height; z = 0 means no face
vec2 warp(vec2 p) {
  if (face.z <= 0. || slim <= 0.) return p;
  vec2 c = face.xy + vec2(0., face.w * .35);            // cheeks and jaw
  vec2 d = (p - c) / vec2(face.z * 1.25, face.w * 1.1);
  float w = 1. - smoothstep(.45, 1.25, length(d));
  p.x += (p.x - c.x) * slim * w;                        // sample further out = cheeks move in
  return p;
}
void main() {
  vec3 c = texture2D(vid, warp(uv)).rgb;
  float y = dot(c, vec3(.299, .587, .114));
  // warm, mid-bright pixels count as skin; a wide soft ramp so the face lightens evenly, no patches
  float skin = smoothstep(.0, .18, c.r - c.b) * smoothstep(.12, .3, y) * (1. - smoothstep(.85, .98, y));
  vec3 w = c + (1. - c) * k * .5;                 // lighter
  w = mix(w, vec3(dot(w, vec3(.299, .587, .114))), k * .22);   // a little less yellow-orange, hue kept
  w.r += k * .015;                                // keep a healthy warmth
  c = mix(c, w, skin);
  gl_FragColor = vec4(c, 1.);
}`;

  function init(canvas) {
    if (gl && cv === canvas) return;
    cv = canvas;
    gl = canvas.getContext('webgl', { antialias: false, alpha: false, premultipliedAlpha: false });
    if (!gl) throw new Error('这个浏览器不支持 WebGL');
    const sh = (t, s) => { const x = gl.createShader(t); gl.shaderSource(x, s); gl.compileShader(x);
      if (!gl.getShaderParameter(x, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(x)); return x; };
    prog = gl.createProgram();
    gl.attachShader(prog, sh(gl.VERTEX_SHADER, VS)); gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, FS));
    gl.bindAttribLocation(prog, 0, 'p'); gl.linkProgram(prog); gl.useProgram(prog);
    uK = gl.getUniformLocation(prog, 'k');
    uFace = gl.getUniformLocation(prog, 'face'); uSlim = gl.getUniformLocation(prog, 'slim');
    gl.uniform1i(gl.getUniformLocation(prog, 'vid'), 0);
    const b = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    tex = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  }

  function frame() {
    if (!running) return;
    const W = video.videoWidth, H = video.videoHeight;
    if (W && H && video.readyState >= 2) {
      if (cv.width !== W || cv.height !== H) { cv.width = W; cv.height = H; }
      gl.viewport(0, 0, W, H);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, video);
      if (slimOn && fd && frameN++ % 3 === 0) findFace();
      gl.uniform1f(uK, LEVELS[level] || 0);
      gl.uniform1f(uSlim, slimOn && fd ? SLIM : 0);
      const f = face || [0, 0, 0, 0];
      gl.uniform4f(uFace, f[0], f[1], f[2], f[3]);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }
    if (video.requestVideoFrameCallback) vfcId = video.requestVideoFrameCallback(frame);
    else rafId = requestAnimationFrame(frame);
  }

  window.Whiten = {
    isRunning: () => running,
    setLevel(n) { level = n; },
    setSlim(on) { slimOn = on; if (!on) face = null; },
    loadFace,
    faceReady: () => !!fd,
    debug: () => ({ face, slimOn, fd: !!fd }),
    start(v, canvas) {
      init(canvas); video = v;
      if (running) return;
      running = true; frame();
    },
    stop() {
      running = false;
      if (video && video.cancelVideoFrameCallback && vfcId) video.cancelVideoFrameCallback(vfcId);
      cancelAnimationFrame(rafId); vfcId = rafId = 0;
    },
  };
})();
