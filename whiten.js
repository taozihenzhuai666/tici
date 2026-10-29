// 美白 + 瘦脸，只动脸（和脖子上露出的皮肤），别的地方（墙、衣服、背景）一点不动，清晰度、对比度不降。
// 找脸：用 MediaPipe 人脸检测（打开网页就开始下载，用 CPU）。放在后台线程（face-worker.js）里跑，画面和录像不等它、不掉帧；
//       后台线程起不来或不回应，就退回页面里跑（普通画布），找得慢一点但照样能用。
//       模型程序从国内镜像下（github.io 在国内太慢），下过一次存在手机里。
//       只信模型：没找到脸、模型还没好，就什么都不改，不会去动墙、抱枕（不再按肤色猜）。
// 脸朝哪边：不靠猜。模型给出眼睛和嘴的位置，从眼睛指向嘴就是下巴方向——手机怎么拿、画面怎么转都对得上，
//       脸是倒着、歪着都能认出来，不会再锁在错的方向上。
// 美白：脸和脖子范围里、而且是皮肤颜色的地方提亮、去黄（暗部、轮廓不变，不发灰）。
// 瘦脸：两颊、下巴往里收，按真正的下巴方向收。
// 用法：Whiten.start(video, canvas); Whiten.setLevel(0..3); Whiten.setSlim(true); Whiten.setTurn(度数); Whiten.loadFace(); Whiten.stop()
(function () {
  const LEVELS = { 1: .3, 2: .5, 3: .7 };
  const SLIM = .09;     // cheeks and jaw move in about 9%
  const here = p => new URL(p, document.baseURI).href;
  let gl = null, cv = null, video = null, prog = null, tex = null, U = {};
  let running = false, level = 2, slimOn = false, turnDeg = 0, vfcId = 0, rafId = 0;
  let fd = null, faceLoading = null, modelState = 'idle', loadErr = '', loadMs = 0, tries = 0;
  // face, in the camera picture's own pixels: centre x, y; chin direction dx, dy (unit); half width, half length
  let face = null, lookId = 0, lastTs = 0, found = 0, looked = 0;

  // The face model's program is 9.4 MB. From github.io in China that can take minutes, which is why it
  // never got going on the phone. So: fetch it from a mirror that's quick in China (same files, same
  // version), fall back to this site and jsdelivr, show how far it has got, and keep it on the phone
  // (Cache Storage) so it's instant next time.
  const MP_VER = '0.10.14';
  const SOURCES = [
    `https://registry.npmmirror.com/@mediapipe/tasks-vision/${MP_VER}/files/wasm/`,
    here('mp/wasm/'),
    `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VER}/wasm/`,
  ];
  let stage = '', pct = 0;
  async function getBytes(name, type, big) {
    const key = here('mp/cache/' + MP_VER + '/' + name);
    let box = null;
    try { box = await caches.open('tici-mp'); const hit = await box.match(key); if (hit) return await hit.blob(); } catch (e) {}
    let lastErr = null;
    for (const base of name.endsWith('.tflite') ? [here('mp/')] : SOURCES) {
      const ctl = new AbortController();
      let stall = setTimeout(() => ctl.abort(), 15000);          // nothing arriving for 15 s: next source
      try {
        const r = await fetch(base + name, { signal: ctl.signal });
        if (!r.ok) throw new Error('HTTP ' + r.status);
        const total = +r.headers.get('content-length') || 0, parts = [];
        let got = 0;
        const rd = r.body.getReader();
        for (;;) {
          const { done, value } = await rd.read();
          if (done) break;
          parts.push(value); got += value.length;
          clearTimeout(stall); stall = setTimeout(() => ctl.abort(), 15000);
          if (big && total) pct = Math.min(99, Math.round(got * 100 / total));
        }
        clearTimeout(stall);
        const blob = new Blob(parts, { type });
        try { if (box) await box.put(key, new Response(blob, { headers: { 'Content-Type': type } })); } catch (e) {}
        return blob;
      } catch (e) { clearTimeout(stall); lastErr = e; pct = 0; }
    }
    throw new Error('下载失败：' + (lastErr && lastErr.name === 'AbortError' ? '网速太慢' : lastErr && lastErr.message));
  }
  // Looking runs in a background thread (a worker), so drawing the picture and the recording never wait
  // on it. If the phone can't start it there, or it stops answering, it runs on the page as before.
  function inWorker(files, model) {
    return new Promise((ok, no) => {
      const w = new Worker(here('face-worker.js?v=2'));
      const wait = new Map();
      let n = 0, up = false;
      const fail = e => { w.terminate(); for (const p of wait.values()) p.no(e); wait.clear(); if (!up) no(e); };
      const timer = setTimeout(() => fail(new Error('后台线程卡住')), 15000);
      w.onerror = e => { clearTimeout(timer); fail(new Error(e.message || '后台线程出错')); };
      w.onmessage = ({ data: m }) => {
        if (m.id) { const p = wait.get(m.id); if (p) { wait.delete(m.id); m.error ? p.no(new Error(m.error)) : p.ok(m.dets); } return; }
        clearTimeout(timer);
        if (m.error) return fail(new Error(m.error));
        up = true;
        ok({ where: '后台线程', stop: () => fail(new Error('停了')),
          detect: (cv, ts) => createImageBitmap(cv).then(img => new Promise((ok2, no2) => {
            const id = ++n; wait.set(id, { ok: ok2, no: no2 });
            w.postMessage({ id, img, ts }, [img]);
            setTimeout(() => { if (wait.delete(id)) no2(new Error('后台线程没回应')); }, 3000);
          })) });
      };
      w.postMessage({ init: { bundle: here('mp/vision_bundle.mjs'), js: files.wasmLoaderPath, wasm: files.wasmBinaryPath, model } });
    });
  }
  async function onPage(mp, files, model) {
    const make = mp.FaceDetector.createFromOptions(files, {
      // an ordinary canvas: Safari 17+ would otherwise get an OffscreenCanvas, which iPhones handle badly
      canvas: document.createElement('canvas'),
      baseOptions: { modelAssetBuffer: new Uint8Array(model), delegate: 'CPU' },
      runningMode: 'VIDEO', minDetectionConfidence: .5,
    });
    const det = await Promise.race([make, new Promise((_, no) => setTimeout(() => no(new Error('启动模型卡住')), 30000))]);
    return { where: '页面', stop() {}, detect: (cv, ts) => Promise.resolve(det.detectForVideo(cv, ts).detections || []) };
  }
  // what's needed to start the model on the page instead; a phone where the worker wouldn't start remembers it
  let made = null, workerErr = '';
  try { if (localStorage.getItem('tici.noWorker')) workerErr = '上次没起来：' + localStorage.getItem('tici.noWorker'); } catch (e) {}
  function loadFace() {
    if (fd) return Promise.resolve();
    if (faceLoading) return faceLoading;
    modelState = 'loading'; loadErr = ''; tries++;
    const t0 = performance.now();
    faceLoading = (async () => {
      if (!made) {
        stage = '加载程序';
        const mp = await import(here('mp/vision_bundle.mjs'));
        const kind = await mp.FilesetResolver.isSimdSupported() ? 'vision_wasm_internal' : 'vision_wasm_nosimd_internal';
        stage = '下载模型';
        const [js, wasm, model] = await Promise.all([
          getBytes(kind + '.js', 'text/javascript'), getBytes(kind + '.wasm', 'application/wasm', true), getBytes('blaze_face_short_range.tflite', 'application/octet-stream')]);
        made = [mp, { wasmLoaderPath: URL.createObjectURL(js), wasmBinaryPath: URL.createObjectURL(wasm) }, await model.arrayBuffer()];
      }
      stage = '启动模型';
      let d = null;
      if (!workerErr && window.Worker && window.createImageBitmap && window.OffscreenCanvas) {
        try { d = await inWorker(made[1], made[2]); }
        catch (e) { workerErr = e.message; try { localStorage.setItem('tici.noWorker', e.message); } catch (x) {} }
      }
      fd = d || await onPage(...made);
      modelState = 'ready'; stage = ''; pct = 0; loadMs = Math.round(performance.now() - t0); face = null;
    })().catch(e => {
      modelState = 'failed'; loadErr = stage + '：' + ((e && e.message) || e);
      if (tries < 3) setTimeout(() => { if (!fd && !faceLoading) loadFace().catch(() => {}); }, 3000);   // try again by itself
      throw e;
    }).finally(() => { faceLoading = null; });
    return faceLoading;
  }
  // the worker went wrong while running: carry on on the page
  function toPage(why) {
    const d = fd; fd = null; face = null; queue = []; cand = null;
    workerErr = why; d.stop();
    loadFace().catch(() => {});
  }

  // follow the face smoothly; jump when it's a new one
  function glide(nf) {
    if (!face || Math.hypot(nf[0] - face[0], nf[1] - face[1]) > nf[4] * 1.5) face = nf.slice();
    else {
      for (let i = 0; i < 6; i++) face[i] += (nf[i] - face[i]) * .6;
      const n = Math.hypot(face[2], face[3]) || 1; face[2] /= n; face[3] /= n;
    }
  }
  function fade() { if (face) { face[4] *= .7; face[5] *= .7; if (face[4] < 6) face = null; } }

  // The model knows upright faces best, so the picture is shrunk (quicker) and turned to stand the person
  // up first. Which turn: the model will also "find" an upside-down face, with the eyes and mouth in the
  // wrong places, so when a face first shows up all four turns are compared and the surest one is kept;
  // after that the kept turn is looked at, and every ~3 s it's checked against upside down.
  const rCv = document.createElement('canvas'), rCtx = rCv.getContext('2d');
  let tryT = 0, misses = 0, locked = false, sinceScan = 0;
  async function detectAt(t) {
    const W = video.videoWidth, H = video.videoHeight;
    const s = Math.min(1, 256 / Math.max(W, H)), w = Math.round(W * s), h = Math.round(H * s);
    const side = t % 180 !== 0, sw = side ? h : w, sh = side ? w : h;
    if (rCv.width !== sw || rCv.height !== sh) { rCv.width = sw; rCv.height = sh; }
    rCtx.setTransform(1, 0, 0, 1, sw / 2, sh / 2); rCtx.rotate(t * Math.PI / 180);
    rCtx.drawImage(video, -w / 2, -h / 2, w, h);
    lastTs = Math.max(lastTs + 1, Math.round(performance.now()));
    looked++;
    let best = null;
    const e0 = epoch;
    const dets = await fd.detect(rCv, lastTs);
    if (e0 !== epoch) return null;                                 // the phone was turned meanwhile
    for (const x of dets)
      if (x.keypoints && x.keypoints.length >= 4 && (!best || x.boundingBox.width > best.boundingBox.width)) best = x;
    if (!best) return null;
    const score = (best.categories && best.categories[0] && best.categories[0].score) || 0;
    return { t, s, sw, sh, W, H, det: best, score };
  }
  const ORDER = [0, 180, 90, 270];
  let scanI = 0, queue = [], cand = null, epoch = 0;
  // One look per step, never several in a row: a look takes ~60 ms on a phone, and the picture (and the
  // recording) stands still while it runs, so 4 looks at once came out as a jerky take. Comparing turns
  // is spread over the next steps instead.
  async function findFaceModel() {
    let hit;
    try {
      if (queue.length) {
        const h = await detectAt(queue.shift());
        if (h && (!cand || h.score > cand.score + .02)) cand = h;
        if (queue.length) return;
        hit = cand; cand = null;
      } else if (!locked) {
        // nobody yet: one way per look, in turn; once something shows up, compare the other three
        const t = (turnDeg + ORDER[scanI++ % 4]) % 360;
        cand = await detectAt(t);
        if (cand) { queue = ORDER.map(r => (turnDeg + r) % 360).filter(x => x !== t); return; }
      } else if (++sinceScan >= 15) {                                  // every ~3 s: check it isn't upside down
        sinceScan = 0; cand = await detectAt(tryT); queue = [(tryT + 180) % 360]; return;
      } else hit = await detectAt(tryT);
    } catch (e) {
      queue = []; cand = null; loadErr = '识别出错：' + e.message;
      if (fd && fd.where === '后台线程') toPage(e.message);
      return fade();
    }
    if (!hit) {
      fade();
      if (locked && ++misses >= 10) { misses = 0; locked = false; scanI = 0; }   // gone ~2 s: look all ways again
      return;
    }
    misses = 0; locked = true; found++; tryT = hit.t;
    const { t, s, sw, sh, W, H, det } = hit;
    // turned-picture pixels -> camera-picture pixels
    const a = -t * Math.PI / 180, ca = Math.cos(a), sa = Math.sin(a);
    const back = (px, py) => {
      const x = px - sw / 2, y = py - sh / 2;
      return [W / 2 + (x * ca - y * sa) / s, H / 2 + (x * sa + y * ca) / s];
    };
    const k = det.keypoints, bb = det.boundingBox;
    const P = i => back(k[i].x * sw, k[i].y * sh);
    const eR = P(0), eL = P(1), mo = P(3);
    let dx = mo[0] - (eR[0] + eL[0]) / 2, dy = mo[1] - (eR[1] + eL[1]) / 2;
    const n = Math.hypot(dx, dy) || 1; dx /= n; dy /= n;
    // the chin points down in the turned picture; eyes and mouth only add the head's tilt (keep it within 40°)
    const dn = [Math.sin(t * Math.PI / 180), Math.cos(t * Math.PI / 180)];
    if (dx * dn[0] + dy * dn[1] < .77) { dx = dn[0]; dy = dn[1]; }
    const hw = bb.width / 2 / s, hb = bb.height / 2 / s;
    const c = back(bb.originX + bb.width / 2, bb.originY + bb.height / 2);
    // the model's box stops at the eyebrows: stretch it up over the forehead
    glide([c[0] - dx * hb * .25, c[1] - dy * hb * .25, dx, dy, hw, hb * 1.25]);
    lastScore = Math.round(hit.score * 100) / 100;
  }
  let lastScore = 0;

  const VS = `attribute vec2 p; varying vec2 uv;
void main() { uv = vec2(p.x * .5 + .5, .5 - p.y * .5); gl_Position = vec4(p, 0., 1.); }`;
  // everything in the camera picture's pixels (origin top-left), so a face at any angle works
  const FS = `precision highp float;
varying vec2 uv; uniform sampler2D vid; uniform float k, slim;
uniform vec2 res;     // picture size
uniform vec4 face;    // centre x, y; half width, half length (pixels); z = 0 means no face
uniform vec2 chin;    // which way the chin points (unit)
const vec3 LUM = vec3(.299, .587, .114);
vec2 warp(vec2 p) {
  if (face.z <= 0. || slim <= 0.) return p;
  vec2 across = vec2(-chin.y, chin.x);                   // cheek to cheek
  vec2 r = p - (face.xy + chin * face.w * .3);           // cheeks and jaw
  float u = dot(r, across) / face.z, v = dot(r, chin) / face.w;
  float w = 1. - smoothstep(.5, 1.3, length(vec2(u / 1.3, v / 1.05)));
  w *= .55 + .45 * smoothstep(-.6, .4, v);              // more at the jaw than at the cheekbones
  return p + across * dot(r, across) * slim * w;         // sample further out = cheeks move in
}
void main() {
  vec2 p = warp(uv * res);
  vec2 q = p / res;
  vec3 c = texture2D(vid, q).rgb;
  if (k > 0. && face.z > 0.) {
    vec2 across = vec2(-chin.y, chin.x);
    vec2 r = p - face.xy;
    float u = dot(r, across) / face.z, v = dot(r, chin) / face.w;
    float inFace = 1. - smoothstep(.85, 1.2, length(vec2(u / 1.2, v / 1.15)));
    // the neck under the chin too, so the face isn't whiter than the neck
    float inNeck = (1. - smoothstep(.75, 1.1, abs(u) / .95)) * (1. - smoothstep(1.9, 2.5, v)) * step(0., v);
    float skin = smoothstep(.01, .12, c.r - c.b) * smoothstep(.06, .2, dot(c, LUM)) * smoothstep(-.02, .04, c.r - c.g);
    float a = max(inFace, inNeck) * skin;
    if (a > 0.) {
      vec3 w = pow(max(c, vec3(0.)), vec3(1. - k * .6));    // brighter mid-tones; black stays black
      w = mix(w, vec3(dot(w, LUM)), k * .25);               // less yellow
      w.b += k * .05 * (1. - w.b);                          // a touch cooler, like fair skin
      w = mix(w, w * w * (3. - 2. * w), .2);                // keep the face's own contrast
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
    for (const n of ['k', 'slim', 'res', 'face', 'chin']) U[n] = gl.getUniformLocation(prog, n);
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
      if (!fd) face = null;                                         // no model (yet): leave the picture alone
      const f = face || [0, 0, 0, 1, 0, 0];
      gl.uniform1f(U.k, LEVELS[level] || 0);
      gl.uniform1f(U.slim, slimOn ? SLIM : 0);
      gl.uniform2f(U.res, W, H);
      gl.uniform2f(U.chin, f[2], f[3]);
      gl.uniform4f(U.face, f[0], f[1], f[4], f[5]);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }
    if (video.requestVideoFrameCallback) vfcId = video.requestVideoFrameCallback(frame);
    else rafId = requestAnimationFrame(frame);
  }

  // looking for the face runs on its own clock, not on every picture frame: ~8 times a second in the
  // worker (quicker while comparing turns); ~3 times on the page, where each look holds the picture up
  let lookGen = 0;
  async function look(g) {
    if (g !== lookGen) return;
    if (fd && video && video.videoWidth && video.readyState >= 2) await findFaceModel();
    if (g === lookGen) lookId = setTimeout(look, queue.length ? 60 : fd && fd.where === '页面' ? 350 : 120, g);
  }

  window.Whiten = {
    isRunning: () => running,
    setLevel(n) { level = n; },
    setSlim(on) { slimOn = on; },
    setTurn(deg) {
      deg = ((deg % 360) + 360) % 360;
      if (deg !== turnDeg) { turnDeg = deg; tryT = deg; misses = 0; locked = false; scanI = 0; queue = []; cand = null; epoch++; face = null; }
    },
    loadFace,
    // what's happening, for the buttons: model 'idle' | 'loading' | 'ready' | 'failed'; face found or not
    status: () => ({ model: modelState, face: !!face, stage, pct }),
    // for the long-press details
    debug: () => ({ model: modelState, where: fd ? fd.where : '', workerErr, stage, pct, err: loadErr, loadMs, tries, looked, found, turn: turnDeg, tryT, score: lastScore, face: face && face.map(v => Math.round(v * 100) / 100),
      size: video ? video.videoWidth + '×' + video.videoHeight : '' }),
    start(v, canvas) {
      init(canvas); video = v;
      if (running) return;
      running = true; face = null; queue = []; cand = null; frame(); look(++lookGen);
    },
    stop() {
      running = false;
      if (video && video.cancelVideoFrameCallback && vfcId) video.cancelVideoFrameCallback(vfcId);
      cancelAnimationFrame(rafId); clearTimeout(lookId); lookGen++; vfcId = rafId = lookId = 0;
    },
  };
})();
