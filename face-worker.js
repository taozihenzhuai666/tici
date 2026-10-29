// 找脸放在后台线程里跑：找一次脸在手机上要几十毫秒，放在页面里跑，画面和录像就会跟着卡一下（掉帧）。
// 页面把缩小、转正后的画面（ImageBitmap）发过来，这里找完把眼睛、嘴、脸框的位置发回去。用法见 whiten.js。
let fd = null;
onmessage = async ({ data: m }) => {
  if (m.init) {
    try {
      const mp = await import(m.init.bundle);
      fd = await mp.FaceDetector.createFromOptions({ wasmLoaderPath: m.init.js, wasmBinaryPath: m.init.wasm }, {
        // MediaPipe only makes its own canvas here for Safari it knows by version number; hand it one
        canvas: new OffscreenCanvas(1, 1),
        baseOptions: { modelAssetBuffer: new Uint8Array(m.init.model), delegate: 'CPU' },
        runningMode: 'VIDEO', minDetectionConfidence: .5,
      });
      postMessage({ ready: true });
    } catch (e) { postMessage({ error: String((e && e.message) || e) }); }
    return;
  }
  try {
    const dets = (fd.detectForVideo(m.img, m.ts).detections || [])
      .map(d => ({ keypoints: d.keypoints, boundingBox: d.boundingBox, categories: d.categories }));
    postMessage({ id: m.id, dets });
  } catch (e) { postMessage({ id: m.id, error: String((e && e.message) || e) }); }
  finally { m.img.close(); }
};
