"use strict";
const LABELS = [
  ["세균성 반점병", "Bacterial spot"],
  ["겹무늬병", "Early blight"],
  ["역병", "Late blight"],
  ["잎곰팡이병", "Leaf Mold"],
  ["셉토리아 잎반점병", "Septoria leaf spot"],
  ["점박이응애 피해", "Spider mites"],
  ["표적반점병", "Target Spot"],
  ["황화잎말림바이러스", "TYLCV"],
  ["토마토 모자이크 바이러스", "TMV"],
  ["정상", "Healthy"]
];
const RECALL = [0.9969, 0.8133, 0.9579, 0.9301, 1, 0.9603, 0.9571, 0.9863, 1, 1];
const MATRIX = [
  [318,0,0,0,0,0,0,1,0,0],
  [0,122,13,0,13,0,2,0,0,0],
  [2,1,273,0,9,0,0,0,0,0],
  [0,1,6,133,0,2,1,0,0,0],
  [0,0,0,0,265,0,0,0,0,0],
  [0,0,0,0,0,242,9,0,0,1],
  [5,0,1,0,3,0,201,0,0,0],
  [7,0,2,0,0,2,0,792,0,0],
  [0,0,0,0,0,0,0,0,56,0],
  [0,0,0,0,0,0,0,0,0,237]
];
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];
const el = (name) => document.getElementById(name);
let session = null;
let imageFile = null;
let busy = false;
const analyzeButton = el("analyzeBtn");
function refreshButton() { analyzeButton.disabled = !session || !imageFile || busy; }
function setStatus(title, kind, message) {
  el("modelStatus").textContent = title;
  el("modelStatus").className = "pill " + kind;
  el("modelMessage").textContent = message;
}
async function loadModel(binary, label) {
  if (!window.ort) throw Error("추론 라이브러리를 불러오지 못했습니다. 인터넷 연결을 확인하세요.");
  session = null;
  refreshButton();
  setStatus("모델 로딩 중", "loading", "ONNX 모델을 읽는 중입니다.");
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/";
  const model = await ort.InferenceSession.create(binary, {executionProviders:["wasm"]});
  if (model.inputNames.length !== 1 || model.outputNames.length !== 1) {
    await model.release();
    throw Error("지원되지 않는 ONNX 구조입니다. TomatoLab의 export_onnx.py로 변환하세요.");
  }
  session = model;
  el("modelFilename").textContent = label;
  setStatus("모델 연결됨", "success", "이제 토마토 잎 사진을 업로드해 분류할 수 있습니다.");
  refreshButton();
}
el("modelFile").addEventListener("change", async (event) => {
  const file = event.target.files && event.target.files[0];
  if (!file) return;
  try {
    if (!file.name.toLowerCase().endsWith(".onnx")) throw Error("ONNX 형식의 파일을 선택하세요.");
    await loadModel(await file.arrayBuffer(), file.name);
  } catch (error) {
    setStatus("연결 실패", "error", error.message);
  }
});
function showError(message) {
  el("resultError").hidden = !message;
  el("resultError").textContent = message || "";
}
async function selectImage(file) {
  if (!file) return;
  if (!file.type.startsWith("image/")) return showError("이미지 파일을 선택해 주세요.");
  if (file.size > 20 * 1024 * 1024) return showError("20MB 이하의 이미지를 사용해 주세요.");
  imageFile = file;
  const preview = el("preview");
  if (preview.dataset.objectUrl) URL.revokeObjectURL(preview.dataset.objectUrl);
  preview.dataset.objectUrl = URL.createObjectURL(file);
  preview.src = preview.dataset.objectUrl;
  preview.hidden = false;
  el("placeholder").hidden = true;
  el("imageFilename").textContent = file.name;
  el("results").hidden = true;
  el("emptyResults").hidden = false;
  showError("");
  refreshButton();
}
el("imageFile").addEventListener("change", (event) => selectImage(event.target.files && event.target.files[0]));
const dropzone = el("dropzone");
dropzone.addEventListener("dragover", (event) => { event.preventDefault(); dropzone.classList.add("dragging"); });
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragging"));
dropzone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropzone.classList.remove("dragging");
  selectImage(event.dataTransfer.files && event.dataTransfer.files[0]);
});
async function prepareInput(file) {
  const bitmap = await createImageBitmap(file);
  const canvas = document.createElement("canvas");
  canvas.width = 256;
  canvas.height = 256;
  const context = canvas.getContext("2d", {willReadFrequently:true});
  context.drawImage(bitmap, 0, 0, 256, 256);
  bitmap.close();
  const data = context.getImageData(16,16,224,224).data;
  const channelSize = 224 * 224;
  const tensor = new Float32Array(3 * channelSize);
  for (let pixel = 0; pixel < channelSize; pixel++) {
    for (let channel = 0; channel < 3; channel++) {
      tensor[channel * channelSize + pixel] = (data[pixel * 4 + channel] / 255 - MEAN[channel]) / STD[channel];
    }
  }
  return new ort.Tensor("float32", tensor, [1,3,224,224]);
}
function renderScores(logits) {
  if (logits.length !== 10 || !logits.every(Number.isFinite)) throw Error("10개 범주와 일치하는 모델을 선택하세요.");
  const maximum = Math.max(...logits);
  const weights = logits.map((score) => Math.exp(score - maximum));
  const denominator = weights.reduce((a,b) => a+b, 0);
  const probs = weights.map((weight) => weight/denominator);
  const order = [...probs.keys()].sort((a,b) => probs[b]-probs[a]);
  const best = order[0];
  el("topLabel").textContent = LABELS[best][0] + " · " + LABELS[best][1];
  el("topScore").textContent = (probs[best] * 100).toFixed(1) + "%";
  const ranking = el("ranking");
  ranking.replaceChildren();
  for (const index of order) {
    const row = document.createElement("div");
    row.className = "rank-row";
    const head = document.createElement("div");
    head.className = "rank-row-head";
    const label = document.createElement("span");
    label.textContent = LABELS[index][0];
    const percent = document.createElement("b");
    percent.textContent = (100 * probs[index]).toFixed(2) + "%";
    head.append(label, percent);
    const track = document.createElement("div");
    track.className = "bar-track";
    const fill = document.createElement("div");
    fill.className = "bar-fill";
    fill.style.width = (100 * probs[index]) + "%";
    track.append(fill);
    row.append(head,track);
    ranking.append(row);
  }
  el("emptyResults").hidden = true;
  el("results").hidden = false;
}
analyzeButton.addEventListener("click", async () => {
  if (busy || !session || !imageFile) return;
  busy = true;
  analyzeButton.textContent = "분석 중...";
  refreshButton();
  showError("");
  try {
    const input = await prepareInput(imageFile);
    const output = await session.run({[session.inputNames[0]]:input});
    renderScores([...output[session.outputNames[0]].data]);
  } catch (error) {
    showError("분류 실패: " + error.message);
  } finally {
    busy = false;
    analyzeButton.textContent = "사진 분석하기 →";
    refreshButton();
  }
});
function makePerformance() {
  for (let i=0;i<10;i++) {
    const row = document.createElement("div");
    row.className = "recall-row";
    const label = document.createElement("span");
    label.textContent = LABELS[i][0];
    const track = document.createElement("div");
    track.className = "bar-track";
    const fill = document.createElement("div");
    fill.className = "bar-fill";
    fill.style.width = (RECALL[i]*100)+"%";
    if (RECALL[i]<.9) fill.style.background = "#e08a59";
    track.append(fill);
    const value = document.createElement("b");
    value.textContent = (RECALL[i]*100).toFixed(1)+"%";
    row.append(label,track,value);
    el("classRecall").append(row);
  }
  const header = document.createElement("tr");
  for(let j=0;j<12;j++) {
    const th = document.createElement("th");
    th.textContent = j===0?"":j===11?"합계":String(j);
    header.append(th);
  }
  el("confusionTable").append(header);
  for(let i=0;i<10;i++) {
    const row = document.createElement("tr");
    const th = document.createElement("th");
    th.textContent = String(i+1);
    row.append(th);
    const total = MATRIX[i].reduce((a,b)=>a+b,0);
    for(let j=0;j<10;j++) {
      const cell = document.createElement("td");
      const count = MATRIX[i][j];
      const intensity = count / Math.max(1,total);
      cell.textContent = String(count);
      cell.style.background = "rgba(37,89,202,"+(0.04+0.88*intensity).toFixed(3)+")";
      cell.style.color = intensity>.45?"#fff":"#40536b";
      cell.title = LABELS[i][0]+" → "+LABELS[j][0]+": "+count+"장";
      row.append(cell);
    }
    const end = document.createElement("th");
    end.textContent = String(total);
    row.append(end);
    el("confusionTable").append(row);
    const legend = document.createElement("span");
    legend.textContent = (i+1)+". "+LABELS[i][0];
    el("matrixLegend").append(legend);
  }
}
makePerformance();
document.querySelectorAll(".nav").forEach((button)=>{
  button.addEventListener("click",()=>{
    const selected = button.dataset.view;
    document.querySelectorAll(".nav").forEach((nav)=>nav.classList.toggle("active",nav===button));
    document.querySelectorAll(".view").forEach((view)=>view.classList.toggle("active",view.id==="view-"+selected));
    el("sidebar").classList.remove("open");
    window.scrollTo({top:0,behavior:"smooth"});
  });
});
el("menuToggle").addEventListener("click",()=>{
  if (window.matchMedia("(max-width: 760px)").matches) el("sidebar").classList.toggle("open");
  else el("sidebar").classList.toggle("collapsed");
});
fetch("./config.json").then((r)=>r.ok?r.json():{}).then(async(config)=>{
  if (!config.modelUrl) return;
  const response = await fetch(config.modelUrl);
  if (!response.ok) throw Error("모델 다운로드에 실패했습니다 (HTTP "+response.status+")");
  await loadModel(await response.arrayBuffer(),"호스팅 모델");
}).catch((error)=>{
  setStatus("모델 연결 필요","neutral","자동 연결 실패: "+error.message+" · ONNX 파일을 직접 선택하세요.");
});