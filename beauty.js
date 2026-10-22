// 美颜：人像分割 + 人脸检测（MediaPipe，文件都在本站 mp/ 目录）+ WebGL 逐帧处理
// 人物：磨皮、提亮、去黄；脸：两颊往里收；背景：模糊 + 压暗（景深）
// 用法：await Beauty.load(); Beauty.start(video, canvas); Beauty.setLevel(1..3); Beauty.stop()
(function () {
  // 每档的参数：磨皮、提亮、美白、瘦脸、背景模糊、背景压暗
  const LEVELS = {
    1: { smooth: .35, light: .05, white: .05, slim: .035, blur: 1.0, dark: .18 },
    2: { smooth: .60, light: .09, white: .10, slim: .06, blur: 1.6, dark: .32 },
    3: { smooth: .80, light: .13, white: .15, slim: .085, blur: 2.2, dark: .45 },
  };
  const here = p => new URL(p, document.baseURI).href;

  let seg = null, fd = null, loading = null;
  let gl = null, cv = null, video = null, running = false, level = 2;
  let P = {}, T = {}, F = {}, small = [0, 0];
  let maskBuf = null, maskW = 0, maskH = 0, hasMask = false, maskFlip = null;
  let face = null, frameN = 0, lastTs = 0, vfcId = 0, rafId = 0;
  // the models only need a small picture; shrinking it first keeps the phone cool
  const sCv = document.createElement('canvas'), sCtx = sCv.getContext('2d', { willReadFrequently: false });

  function load() {
    if (loading) return loading;
    loading = (async () => {
      const mp = await import(here('mp/vision_bundle.mjs'));
      const files = await mp.FilesetResolver.forVisionTasks(here('mp/wasm'));
      const make = async (Cls, model, extra) => {
        let err;
        for (const delegate of ['GPU', 'CPU']) {
          try {
            return await Cls.createFromOptions(files, Object.assign({
              baseOptions: { modelAssetPath: here(model), delegate }, runningMode: 'VIDEO',
            }, extra));
          } catch (e) { err = e; }
        }
        throw err;
      };
      seg = await make(mp.ImageSegmenter, 'mp/selfie_segmenter.tflite', { outputCategoryMask: false, outputConfidenceMasks: true });
      try { fd = await make(mp.FaceDetector, 'mp/blaze_face_short_range.tflite', { minDetectionConfidence: .5 }); } catch (e) { fd = null; }
    })();
    loading.catch(() => { loading = null; });
    return loading;
  }

  // ---------- WebGL ----------
  const VS = `#version 300 es
in vec2 p; out vec2 uv;
void main() { uv = p * .5 + .5; gl_Position = vec4(p, 0., 1.); }`;

  const BLUR = `#version 300 es
precision mediump float;
in vec2 uv; out vec4 o;
uniform sampler2D src; uniform vec2 dir;
void main() {
  float w[5] = float[](.227027, .1945946, .1216216, .054054, .016216);
  vec3 c = texture(src, uv).rgb * w[0];
  for (int i = 1; i < 5; i++) {
    c += texture(src, uv + dir * float(i)).rgb * w[i];
    c += texture(src, uv - dir * float(i)).rgb * w[i];
  }
  o = vec4(c, 1.);
}`;

  const FINAL = `#version 300 es
precision highp float;
in vec2 uv; out vec4 o;
uniform sampler2D vid, msk, bg;
uniform vec2 px;          // one pixel in uv
uniform vec4 face;        // centre x, y, half width, half height (uv, origin bottom-left); z = 0 means no face
uniform float slim, smoothK, light, white, dark, hasMask, maskFlip;

vec2 warp(vec2 p) {
  if (face.z <= 0.) return p;
  vec2 c = face.xy - vec2(0., face.w * .35);           // centre on cheeks / jaw
  vec2 d = (p - c) / vec2(face.z * 1.25, face.w * 1.1);
  float w = 1. - smoothstep(.45, 1.25, length(d));
  p.x += (p.x - c.x) * slim * w;                        // sample further out = cheeks move in
  return p;
}
float maskAt(vec2 q) {
  if (hasMask < .5) return 1.;
  vec2 m = vec2(q.x, maskFlip > .5 ? 1. - q.y : q.y);
  return smoothstep(.55, .9, texture(msk, m).r);   // tight edge: no bright rim of wall around the head
}
void main() {
  vec2 q = warp(uv);
  vec3 c0 = texture(vid, q).rgb;
  // edge-keeping blur for skin
  vec3 sum = c0; float ws = 1.;
  for (int i = 0; i < 12; i++) {
    float a = float(i) * .5235988;
    float r = i < 6 ? 2.5 : 5.;
    vec2 off = vec2(cos(a), sin(a)) * px * r * 1.6;
    vec3 c = texture(vid, q + off).rgb;
    vec3 dc = c - c0;
    float w = exp(-dot(dc, dc) * 90.);
    sum += c * w; ws += w;
  }
  vec3 sm = sum / ws;
  float skin = smoothstep(.02, .12, c0.r - c0.b) * step(c0.b, c0.r);
  float m = maskAt(q);
  vec3 per = mix(c0, sm, smoothK * skin * m);
  per.b += white * .45 * max(0., per.r - per.b);        // less yellow
  per.g += white * .15 * max(0., per.r - per.g);
  per = per + (1. - per) * (light + white * .3);        // lift
  vec3 b = texture(bg, uv).rgb * (1. - dark);
  o = vec4(mix(b, per, m), 1.);
}`;

  function prog(fs) {
    const mk = (type, src) => {
      const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
      return s;
    };
    const p = gl.createProgram();
    gl.attachShader(p, mk(gl.VERTEX_SHADER, VS)); gl.attachShader(p, mk(gl.FRAGMENT_SHADER, fs));
    gl.bindAttribLocation(p, 0, 'p'); gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    const u = {}; const n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
    for (let i = 0; i < n; i++) { const a = gl.getActiveUniform(p, i); u[a.name] = gl.getUniformLocation(p, a.name); }
    return { p, u };
  }
  function texture() {
    const t = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return t;
  }
  function target(w, h) {
    const t = texture();
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    const f = gl.createFramebuffer(); gl.bindFramebuffer(gl.FRAMEBUFFER, f);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t, 0);
    return { t, f, w, h };
  }
  function initGL(canvas) {
    if (gl && cv === canvas) return;
    cv = canvas;
    gl = canvas.getContext('webgl2', { premultipliedAlpha: false, antialias: false, alpha: false });
    if (!gl) throw new Error('这个浏览器不支持 WebGL2');
    P.blur = prog(BLUR); P.fin = prog(FINAL);
    const b = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    T.vid = texture(); T.msk = texture();
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, 1, 1, 0, gl.RED, gl.UNSIGNED_BYTE, new Uint8Array([255]));
    small = [0, 0];
  }
  function sizeTo(w, h) {
    if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
    const sw = Math.max(64, Math.round(w / 4)), sh = Math.max(64, Math.round(h / 4));
    if (small[0] !== sw || small[1] !== sh) {
      if (F.a) { gl.deleteFramebuffer(F.a.f); gl.deleteTexture(F.a.t); gl.deleteFramebuffer(F.b.f); gl.deleteTexture(F.b.t); }
      F.a = target(sw, sh); F.b = target(sw, sh); small = [sw, sh];
    }
  }
  function blurPass(srcTex, dst, dx, dy) {
    gl.bindFramebuffer(gl.FRAMEBUFFER, dst.f); gl.viewport(0, 0, dst.w, dst.h);
    gl.useProgram(P.blur.p);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, srcTex);
    gl.uniform1i(P.blur.u.src, 0); gl.uniform2f(P.blur.u.dir, dx, dy);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  // ---------- per frame ----------
  function analyse(ts) {
    const W = video.videoWidth, H = video.videoHeight;
    const sw = 256, sh = Math.max(64, Math.round(256 * H / W));
    if (sCv.width !== sw || sCv.height !== sh) { sCv.width = sw; sCv.height = sh; }
    sCtx.drawImage(video, 0, 0, sw, sh);
    try {
      const r = seg.segmentForVideo(sCv, ts);
      const m = r.confidenceMasks && r.confidenceMasks[0];
      if (m) {
        const f = m.getAsFloat32Array();
        if (!maskBuf || maskBuf.length !== f.length) { maskBuf = new Float32Array(f.length); maskBuf.set(f); }
        else for (let i = 0; i < f.length; i++) maskBuf[i] += (f[i] - maskBuf[i]) * .6;   // calm the edges between frames
        maskW = m.width; maskH = m.height;
        const u8 = new Uint8Array(maskBuf.length);
        for (let i = 0; i < maskBuf.length; i++) u8[i] = maskBuf[i] * 255;
        gl.bindTexture(gl.TEXTURE_2D, T.msk);
        gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
        gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, maskW, maskH, 0, gl.RED, gl.UNSIGNED_BYTE, u8);
        hasMask = true;
      }
      r.close();
    } catch (e) { /* keep last mask */ }
    if (fd && frameN % 3 === 0) {
      try {
        const d = fd.detectForVideo(sCv, ts).detections || [];
        let best = null;
        for (const x of d) if (!best || x.boundingBox.width > best.boundingBox.width) best = x;
        if (best) {
          const bb = best.boundingBox;
          const nf = [(bb.originX + bb.width / 2) / sw, 1 - (bb.originY + bb.height / 2) / sh, bb.width / 2 / sw, bb.height / 2 / sh];
          face = face ? face.map((v, i) => v + (nf[i] - v) * .4) : nf;
        } else if (face) {
          face[2] *= .8; face[3] *= .8; if (face[2] < .01) face = null;   // fade out when the face leaves
        }
      } catch (e) {}
    }
  }

  function frame() {
    if (!running) return;
    const W = video.videoWidth, H = video.videoHeight;
    if (W && H && video.readyState >= 2) {
      sizeTo(W, H);
      const ts = Math.max(lastTs + 1, Math.round(performance.now()));
      lastTs = ts;
      if (seg) analyse(ts);
      frameN++;
      const L = LEVELS[level] || LEVELS[2];
      gl.bindTexture(gl.TEXTURE_2D, T.vid);
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, video);
      // background: shrink + blur twice
      const sx = L.blur / small[0], sy = L.blur / small[1];
      blurPass(T.vid, F.a, sx, 0); blurPass(F.a.t, F.b, 0, sy);
      blurPass(F.b.t, F.a, sx * 1.8, 0); blurPass(F.a.t, F.b, 0, sy * 1.8);
      // final
      gl.bindFramebuffer(gl.FRAMEBUFFER, null); gl.viewport(0, 0, W, H);
      gl.useProgram(P.fin.p); const u = P.fin.u;
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, T.vid); gl.uniform1i(u.vid, 0);
      gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, T.msk); gl.uniform1i(u.msk, 1);
      gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, F.b.t); gl.uniform1i(u.bg, 2);
      gl.uniform2f(u.px, 1 / W, 1 / H);
      gl.uniform4f(u.face, ...(face || [0, 0, 0, 0]));
      gl.uniform1f(u.slim, L.slim); gl.uniform1f(u.smoothK, L.smooth);
      gl.uniform1f(u.light, L.light); gl.uniform1f(u.white, L.white); gl.uniform1f(u.dark, hasMask ? L.dark : 0);
      gl.uniform1f(u.hasMask, hasMask ? 1 : 0); gl.uniform1f(u.maskFlip, maskFlip === false ? 0 : 1);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }
    schedule();
  }
  function schedule() {
    if (!running) return;
    if (video.requestVideoFrameCallback) vfcId = video.requestVideoFrameCallback(frame);
    else rafId = requestAnimationFrame(frame);
  }

  window.Beauty = {
    load,
    isReady: () => !!seg,
    isRunning: () => running,
    setLevel(n) { level = n; },
    start(v, canvas) {
      initGL(canvas);
      video = v;
      if (running) return;
      running = true; maskBuf = null; hasMask = false; face = null;
      schedule();
    },
    stop() {
      running = false;
      if (video && video.cancelVideoFrameCallback && vfcId) video.cancelVideoFrameCallback(vfcId);
      cancelAnimationFrame(rafId); vfcId = rafId = 0;
    },
    // for tests: what the models currently see
    debug: () => ({ hasMask, maskW, maskH, face, frameN }),
    _setMaskFlip(v) { maskFlip = v; },
  };
})();
