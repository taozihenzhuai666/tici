// 美白 + 瘦脸，只动脸，别的地方（墙、衣服、背景）一点不动，清晰度、对比度不降。
// 找脸：用 MediaPipe 人脸检测（文件在本站 mp/，开美白或瘦脸时加载，用 CPU——iPhone 上 GPU 会卡住不动）。
//       模型能用时只信模型：没找到脸就什么都不改，不会去动墙、抱枕。
//       模型实在加载不了，才用保底办法按肤色找脸，而且只做美白，不做瘦脸（找错了扭曲很难看）。
// 手机横拿时人在画面里是躺着的：先把画面转正再找脸，结果再转回来；转哪个方向找到脸就一直用那个方向。
// 美白：只在脸的椭圆里、而且是皮肤颜色的地方提亮中间调（暗部、轮廓不变，不发灰）。
// 瘦脸：两颊、下巴往里收一点点，按真正的下巴方向收。
// 用法：Whiten.start(video, canvas); Whiten.setLevel(0..3); Whiten.setSlim(true); Whiten.setTurn(度数); Whiten.loadFace(); Whiten.stop()
(function () {
  const LEVELS = { 1: .2, 2: .35, 3: .5 };
  const SLIM = .06;    // "a little": cheeks move in about 6%
  const LOAD_LIMIT = 30000;
  const here = p => new URL(p, document.baseURI).href;
  let gl = null, cv = null, video = null, prog = null, uK = null, uFace = null, uSlim = null, uChin = null, tex = null;
  let running = false, level = 2, slimOn = false, turnDeg = 0, vfcId = 0, rafId = 0;
  let fd = null, faceLoading = null, modelState = 'idle', face = null, faceBy = '', faceT = 0, frameN = 0, lastTs = 0;
  const gCv = document.createElement('canvas'), gCtx = gCv.getContext('2d', { willReadFrequently: true });

  function loadFace() {
    if (fd) return Promise.resolve();
    if (faceLoading) return faceLoading;
    modelState = 'loading';
    const load = (async () => {
      const mp = await import(here('mp/vision_bundle.mjs'));
      const files = await mp.FilesetResolver.forVisionTasks(here('mp/wasm'));
      fd = await mp.FaceDetector.createFromOptions(files, {
        baseOptions: { modelAssetPath: here('mp/blaze_face_short_range.tflite'), delegate: 'CPU' },
        runningMode: 'VIDEO', minDetectionConfidence: .5,
      });
    })();
    const limit = new Promise((_, no) => setTimeout(() => no(new Error('加载超时')), LOAD_LIMIT));
    faceLoading = Promise.race([load, limit]).then(
      () => { modelState = 'ready'; face = null; },
      e => { faceLoading = null; if (!fd) modelState = 'failed'; throw e; });   // tried again next time the camera opens
    return faceLoading;
  }

  // t: how far the picture was turned to stand the face up (says which way the chin points)
  function glide(nf, by, t) {
    if (!face || faceBy !== by || faceT !== t) face = nf;
    else face = face.map((v, i) => v + (nf[i] - v) * .4);
    faceBy = by; faceT = t;
  }
  function fade() { if (face) { face[2] *= .85; face[3] *= .85; if (face[2] < .02) face = null; } }

  // The model only knows upright faces. The picture is shrunk (quicker) and stood up first, and the box
  // turned back afterwards. Which way is up isn't certain (mirroring, how Safari hands the picture over):
  // the expected turn is tried first; once a turn has found a face it's kept unless the face is gone ~2 s.
  const rCv = document.createElement('canvas'), rCtx = rCv.getContext('2d');
  const FLIPS = [0, 180, 90, 270];
  let flip = 0, misses = 0, locked = false;
  function findFaceModel() {
    const W = video.videoWidth, H = video.videoHeight;
    const t = ((turnDeg + FLIPS[flip]) % 360 + 360) % 360;
    const s = Math.min(1, 480 / Math.max(W, H)), w = Math.round(W * s), h = Math.round(H * s);
    const side = t % 180 !== 0, sw = side ? h : w, sh = side ? w : h;
    if (rCv.width !== sw || rCv.height !== sh) { rCv.width = sw; rCv.height = sh; }
    rCtx.setTransform(1, 0, 0, 1, sw / 2, sh / 2); rCtx.rotate(t * Math.PI / 180);
    rCtx.drawImage(video, -w / 2, -h / 2, w, h);
    lastTs = Math.max(lastTs + 1, Math.round(performance.now()));
    let d;
    try { d = fd.detectForVideo(rCv, lastTs).detections || []; } catch (e) { return fade(); }
    let best = null;
    for (const x of d) if (!best || x.boundingBox.width > best.boundingBox.width) best = x;
    if (!best) {
      fade();
      if (++misses >= (locked ? 20 : 3)) { misses = 0; locked = false; flip = (flip + 1) % 4; }
      return;
    }
    misses = 0; locked = true;
    const bb = best.boundingBox;
    // the model's box stops at the eyebrows: stretch it up over the forehead (upright terms)
    const hh = bb.height / 2 / sh;
    const a = (bb.originX + bb.width / 2) / sw, b = (bb.originY + bb.height / 2) / sh - hh * .25;
    const hw = bb.width / 2 / sw, hv = hh * 1.25;
    // back to the camera picture's own terms
    if (t === 90) glide([b, 1 - a, hv, hw], 'model', t);
    else if (t === 180) glide([1 - a, 1 - b, hw, hv], 'model', t);
    else if (t === 270) glide([1 - b, a, hv, hw], 'model', t);
    else glide([a, b, hw, hv], 'model', t);
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
    if (best) glide(best, 'skin', turnDeg); else fade();
  }

  const VS = `attribute vec2 p; varying vec2 uv;
void main() { uv = vec2(p.x * .5 + .5, .5 - p.y * .5); gl_Position = vec4(p, 0., 1.); }`;
  const FS = `precision highp float;
varying vec2 uv; uniform sampler2D vid; uniform float k, slim;
uniform vec4 face;   // centre x, y (origin top-left), half width, half height; z = 0 means no face
uniform vec2 chin;   // which way the chin points in the picture: (0,1) upright, (1,0) / (-1,0) lying, (0,-1) upside down
const vec3 LUM = vec3(.299, .587, .114);
vec2 warp(vec2 p) {
  if (face.z <= 0. || slim <= 0.) return p;
  vec2 across = vec2(abs(chin.y), abs(chin.x));          // cheek to cheek
  float hl = dot(abs(chin), face.zw), hw = dot(across, face.zw);   // half length, half width of the face
  vec2 c = face.xy + chin * hl * .35;                    // cheeks and jaw
  vec2 r = p - c;
  float u = dot(r, across), v = dot(r, chin);
  float w = 1. - smoothstep(.45, 1.25, length(vec2(u / (hw * 1.25), v / (hl * 1.1))));
  return p + across * u * slim * w;                     // sample further out = cheeks move in
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
    uFace = gl.getUniformLocation(prog, 'face'); uSlim = gl.getUniformLocation(prog, 'slim'); uChin = gl.getUniformLocation(prog, 'chin');
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

  const CHIN = { 0: [0, 1], 90: [1, 0], 180: [0, -1], 270: [-1, 0] };
  function frame() {
    if (!running) return;
    const W = video.videoWidth, H = video.videoHeight;
    if (W && H && video.readyState >= 2) {
      if (cv.width !== W || cv.height !== H) { cv.width = W; cv.height = H; }
      gl.viewport(0, 0, W, H);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, video);
      if (frameN++ % 3 === 0) {
        if (fd) findFaceModel();
        else if (modelState === 'failed' && level) findFaceSkin();   // last resort, whitening only
        else face = null;                                           // model still loading: leave the picture alone
      }
      const f = face || [0, 0, 0, 0], ch = CHIN[faceT] || CHIN[0];
      gl.uniform1f(uK, LEVELS[level] || 0);
      gl.uniform1f(uSlim, slimOn && faceBy === 'model' ? SLIM : 0);
      gl.uniform2f(uChin, ch[0], ch[1]);
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
    setTurn(deg) {
      deg = ((deg % 360) + 360) % 360;
      if (deg !== turnDeg) { turnDeg = deg; flip = 0; misses = 0; locked = false; face = null; }
    },
    loadFace,
    // what's happening, for the slim button: model 'idle' | 'loading' | 'ready' | 'failed'; face found or not
    status: () => ({ model: modelState, face: !!face, by: faceBy }),
    debug: () => ({ face, by: faceBy, chinTurn: faceT, model: modelState }),
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
