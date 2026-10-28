// 美白 + 瘦脸，只动脸，别的地方（墙、衣服、背景）一点不动，清晰度、对比度不降。
// 找脸：有 MediaPipe 人脸检测就用它（文件在本站 mp/，打开瘦脸时才加载）；
//       加载不了时用保底办法——在一张很小的画面上按肤色找出最像脸的一块。
// 美白：只在脸的椭圆里、而且是皮肤颜色的地方提亮中间调（暗部、轮廓不变，不发灰）。
// 瘦脸：两颊、下巴往里收一点点。
// 用法：Whiten.start(video, canvas); Whiten.setLevel(0..3); Whiten.setSlim(true); Whiten.loadFace(); Whiten.stop()
(function () {
  const LEVELS = { 1: .2, 2: .35, 3: .5 };
  const SLIM = .06;    // "a little": cheeks move in about 6%
  const here = p => new URL(p, document.baseURI).href;
  let gl = null, cv = null, video = null, prog = null, uK = null, uFace = null, uSlim = null, uSide = null, tex = null;
  let running = false, level = 2, slimOn = false, sideways = false, turnDeg = 0, vfcId = 0, rafId = 0;
  let fd = null, faceLoading = null, face = null, frameN = 0, lastTs = 0;
  const gCv = document.createElement('canvas'), gCtx = gCv.getContext('2d', { willReadFrequently: true });

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

  function glide(nf) { face = face ? face.map((v, i) => v + (nf[i] - v) * .4) : nf; }
  function fade() { if (face) { face[2] *= .85; face[3] *= .85; if (face[2] < .02) face = null; } }

  // with the model. It only knows upright faces, so when the person lies across the picture
  // (phone held sideways) the picture is stood up first and the box turned back afterwards.
  // Which way is up isn't certain (mirroring, how Safari hands the picture over), so when nothing is
  // found the next quarter turn is tried; the turn that finds a face is kept.
  const rCv = document.createElement('canvas'), rCtx = rCv.getContext('2d');
  let flip = 0, misses = 0, lastT = 0;
  const FLIPS = [0, 180, 90, 270];
  function findFaceModel() {
    const W = video.videoWidth, H = video.videoHeight;
    const t = lastT = ((turnDeg + FLIPS[flip]) % 360 + 360) % 360;
    let src = video, sw = W, sh = H;
    if (t) {
      const s = Math.min(1, 640 / Math.max(W, H)), w = Math.round(W * s), h = Math.round(H * s);
      const side = t % 180 !== 0;
      sw = side ? h : w; sh = side ? w : h;
      if (rCv.width !== sw || rCv.height !== sh) { rCv.width = sw; rCv.height = sh; }
      rCtx.setTransform(1, 0, 0, 1, sw / 2, sh / 2); rCtx.rotate(t * Math.PI / 180);
      rCtx.drawImage(video, -w / 2, -h / 2, w, h);
      src = rCv;
    }
    lastTs = Math.max(lastTs + 1, Math.round(performance.now()));
    try {
      const d = fd.detectForVideo(src, lastTs).detections || [];
      let best = null;
      for (const x of d) if (!best || x.boundingBox.width > best.boundingBox.width) best = x;
      if (!best) { if (++misses >= 2) { misses = 0; flip = (flip + 1) % 4; } return findFaceSkin(); }
      misses = 0;
      const bb = best.boundingBox;
      // the model's box stops at the eyebrows: stretch it up over the forehead (upright terms)
      const hh = bb.height / 2 / sh;
      const a = (bb.originX + bb.width / 2) / sw, b = (bb.originY + bb.height / 2) / sh - hh * .25;
      const hw = bb.width / 2 / sw, hv = hh * 1.25;
      // back to the camera picture's own terms
      if (t === 90) glide([b, 1 - a, hv, hw]);
      else if (t === 180) glide([1 - a, 1 - b, hw, hv]);
      else if (t === 270) glide([1 - b, a, hv, hw]);
      else glide([a, b, hw, hv]);
    } catch (e) { findFaceSkin(); }
  }

  // without the model: the biggest solid skin-coloured blob that isn't a whole wall
  function findFaceSkin() {
    const W = video.videoWidth, H = video.videoHeight;
    const gw = 48, gh = Math.max(16, Math.round(48 * H / W)), n = gw * gh;
    if (gCv.width !== gw || gCv.height !== gh) { gCv.width = gw; gCv.height = gh; }
    gCtx.drawImage(video, 0, 0, gw, gh);
    const px = gCtx.getImageData(0, 0, gw, gh).data;
    const m = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      const r = px[i * 4], g = px[i * 4 + 1], b = px[i * 4 + 2];
      const y = .299 * r + .587 * g + .114 * b;
      const cb = 128 - .168736 * r - .331264 * g + .5 * b, cr = 128 + .5 * r - .418688 * g - .081312 * b;
      m[i] = cr >= 136 && cr <= 175 && cb >= 77 && cb <= 127 && y > 45 && y < 240 && r > g && r - b > 12 ? 1 : 0;
    }
    const seen = new Uint8Array(n), stack = [];
    let best = null, bestScore = 0;
    for (let s0 = 0; s0 < n; s0++) {
      if (!m[s0] || seen[s0]) continue;
      let area = 0, x0 = gw, x1 = 0, y0 = gh, y1 = 0;
      stack.push(s0); seen[s0] = 1;
      while (stack.length) {
        const i = stack.pop(), x = i % gw, y = (i / gw) | 0;
        area++; if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
        if (x > 0 && m[i - 1] && !seen[i - 1]) { seen[i - 1] = 1; stack.push(i - 1); }
        if (x < gw - 1 && m[i + 1] && !seen[i + 1]) { seen[i + 1] = 1; stack.push(i + 1); }
        if (y > 0 && m[i - gw] && !seen[i - gw]) { seen[i - gw] = 1; stack.push(i - gw); }
        if (y < gh - 1 && m[i + gw] && !seen[i + gw]) { seen[i + gw] = 1; stack.push(i + gw); }
      }
      const edges = (x0 === 0) + (x1 === gw - 1) + (y0 === 0) + (y1 === gh - 1);
      if (area < n * .01 || area > n * .45 || edges >= 3) continue;        // too small, or a wall
      if (area / ((x1 - x0 + 1) * (y1 - y0 + 1)) < .35) continue;            // faces are solid, not stringy
      const cx = (x0 + x1 + 1) / 2 / gw, cy = (y0 + y1 + 1) / 2 / gh;
      const score = area * (1.3 - Math.abs(cx - .5)) * (1.3 - Math.abs(cy - .45));   // prefer the middle
      if (score > bestScore) { bestScore = score; best = [cx, cy, (x1 - x0 + 1) / 2 / gw, (y1 - y0 + 1) / 2 / gh]; }
    }
    if (best) glide(best); else fade();
  }

  const VS = `attribute vec2 p; varying vec2 uv;
void main() { uv = vec2(p.x * .5 + .5, .5 - p.y * .5); gl_Position = vec4(p, 0., 1.); }`;
  const FS = `precision highp float;
varying vec2 uv; uniform sampler2D vid; uniform float k, slim, side;   // side: the person lies across the picture (phone turned, picture not)
uniform vec4 face;   // centre x, y (origin top-left), half width, half height; z = 0 means no face
const vec3 LUM = vec3(.299, .587, .114);
vec2 warp(vec2 p) {
  if (face.z <= 0. || slim <= 0.) return p;
  // lying across the picture: work as if upright, and aim at the middle of the face (which end is the chin isn't known)
  vec4 f = side > 0. ? face.yxwz : face;
  if (side > 0.) p = p.yx;
  vec2 c = f.xy + vec2(0., f.w * .35 * (1. - side));      // cheeks and jaw
  vec2 d = (p - c) / vec2(f.z * 1.25, f.w * 1.1);
  float w = 1. - smoothstep(.45, 1.25, length(d));
  p.x += (p.x - c.x) * slim * w;                        // sample further out = cheeks move in
  return side > 0. ? p.yx : p;
}
void main() {
  vec2 q = warp(uv);
  vec3 c = texture2D(vid, q).rgb;
  if (k > 0. && face.z > 0.) {
    vec2 d = (q - face.xy) / vec2(face.z * 1.15, face.w * 1.2);
    float inFace = 1. - smoothstep(.8, 1.1, length(d));
    float skin = smoothstep(.0, .15, c.r - c.b) * smoothstep(.06, .22, dot(c, LUM));
    float a = inFace * skin;
    if (a > 0.) {
      vec3 w = pow(max(c, vec3(0.)), vec3(1. - k * .55));   // brighter mid-tones; black stays black
      w = mix(w, vec3(dot(w, LUM)), k * .18);               // a little less yellow
      w = mix(w, w * w * (3. - 2. * w), .25);               // keep the face's own contrast
      c = mix(c, w, a);
    }
  }
  gl_FragColor = vec4(c, 1.);
}`;

  function init(canvas) {
    if (gl && cv === canvas) return;
    cv = canvas;
    gl = canvas.getContext('webgl', { antialias: false, alpha: false, premultipliedAlpha: false, preserveDrawingBuffer: true });   // kept after drawing, so the turned recording can copy it
    if (!gl) throw new Error('这个浏览器不支持 WebGL');
    const sh = (t, s) => { const x = gl.createShader(t); gl.shaderSource(x, s); gl.compileShader(x);
      if (!gl.getShaderParameter(x, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(x)); return x; };
    prog = gl.createProgram();
    gl.attachShader(prog, sh(gl.VERTEX_SHADER, VS)); gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, FS));
    gl.bindAttribLocation(prog, 0, 'p'); gl.linkProgram(prog); gl.useProgram(prog);
    uK = gl.getUniformLocation(prog, 'k');
    uFace = gl.getUniformLocation(prog, 'face'); uSlim = gl.getUniformLocation(prog, 'slim'); uSide = gl.getUniformLocation(prog, 'side');
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
      if ((level || slimOn) && frameN++ % 3 === 0) (fd ? findFaceModel : findFaceSkin)();
      gl.uniform1f(uK, LEVELS[level] || 0);
      gl.uniform1f(uSlim, slimOn ? SLIM : 0);
      gl.uniform1f(uSide, sideways ? 1 : 0);
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
    setSlim(on) { slimOn = on; },
    setSideways(on) { sideways = on; },
    setTurn(deg) { turnDeg = ((deg % 360) + 360) % 360; sideways = turnDeg % 180 !== 0; flip = 0; misses = 0; },
    loadFace,
    faceReady: () => !!fd,
    debug: () => ({ face, slimOn, fd: !!fd, loading: !!faceLoading && !fd, angle: lastT }),
    start(v, canvas) {
      init(canvas); video = v;
      if (running) return;
      running = true; face = null; frame();
    },
    stop() {
      running = false;
      if (video && video.cancelVideoFrameCallback && vfcId) video.cancelVideoFrameCallback(vfcId);
      cancelAnimationFrame(rafId); vfcId = rafId = 0;
    },
  };
})();
