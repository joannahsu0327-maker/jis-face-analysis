import React, { useEffect, useMemo, useRef, useState } from "react";

let FaceLandmarker = null;
let FilesetResolver = null;

async function loadMediapipe() {
  if (FaceLandmarker && FilesetResolver) return;
  const mod = await import("@mediapipe/tasks-vision");
  FaceLandmarker = mod.FaceLandmarker;
  FilesetResolver = mod.FilesetResolver;
}

const API_BASE_URL =
  "https://script.google.com/macros/s/AKfycbwk6ZFFQgA5z2Ct-YUKPjGj50dK9OfP-sDzoW4TEdh91OIgjXOnayiZ5sRpbqWRxiZYww/exec";

// AI 按鈕統一走 GAS 後端，不在前端放 Gemini API Key。

let faceLandmarkerInstance = null;

async function getFaceLandmarker() {
  if (faceLandmarkerInstance) return faceLandmarkerInstance;
  await loadMediapipe();

  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
  );

  // 使用 CPU 比 GPU 穩。部分預覽環境的 WebGL / GPU delegate 會啟動失敗。
  try {
    faceLandmarkerInstance = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
        delegate: "CPU",
      },
      runningMode: "IMAGE",
      numFaces: 1,
    });
  } catch (error) {
    throw new Error("MediaPipe 點位模型載入失敗，請重新整理頁面後再試；若仍失敗，請改用正式 Vercel 網站測試。" );
  }

  return faceLandmarkerInstance;
}

function distance2D(a, b, width, height) {
  if (!a || !b) return 0;
  const dx = (a.x - b.x) * width;
  const dy = (a.y - b.y) * height;
  return Math.sqrt(dx * dx + dy * dy);
}

function estimateMeasurementsFromLandmarks(landmarks, width, height) {
  const foreheadPoints = [10, 338, 297].map((i) => landmarks[i]).filter(Boolean);
  const top = {
    x: foreheadPoints.reduce((sum, p) => sum + p.x, 0) / foreheadPoints.length,
    y: foreheadPoints.reduce((sum, p) => sum + p.y, 0) / foreheadPoints.length,
  };

  const chin = landmarks[152];
  const left = landmarks[234];
  const right = landmarks[454];
  const browLeft = landmarks[105];
  const browRight = landmarks[334];
  const browCenter = {
    x: (browLeft.x + browRight.x) / 2,
    y: (browLeft.y + browRight.y) / 2,
  };
  const noseBase = landmarks[2];

  const faceLength = distance2D(top, chin, width, height);
  const faceWidth = distance2D(left, right, width, height);
  const upperThird = distance2D(top, browCenter, width, height);
  const middleThird = distance2D(browCenter, noseBase, width, height);
  const lowerThird = distance2D(noseBase, chin, width, height);

  return {
    faceLength: faceLength ? faceLength.toFixed(1) : "",
    faceWidth: faceWidth ? faceWidth.toFixed(1) : "",
    upperThird: upperThird ? upperThird.toFixed(1) : "",
    middleThird: middleThird ? middleThird.toFixed(1) : "",
    lowerThird: lowerThird ? lowerThird.toFixed(1) : "",
  };
}

function loadImageFromBase64(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });
}

const SHEETS = {
  face: "03 臉型 FS",
  ratio: "04 比例 RT",
  line: "05 直曲 LC",
  volume: "06 量感 VM",
  style: "07 風格 ST",
  age: "08 年齡 AG",
};

const steps = [
  { id: "basic", label: "基本資料", icon: "👤" },
  { id: "face", label: "臉型 FS", icon: "🔷" },
  { id: "ratio", label: "比例 RT", icon: "📏" },
  { id: "age", label: "年齡 AG", icon: "🌱" },
  { id: "line", label: "直曲 LC", icon: "〰️" },
  { id: "volume", label: "量感 VM", icon: "◐" },
  { id: "style", label: "風格 ST", icon: "✦" },
  { id: "recommend", label: "建議輸出", icon: "📄" },
];

const fallbackOptions = {
  face: [
    { code: "FS01", name: "圓臉(柔)", feature: "臉長與臉寬接近，輪廓圓潤。", direction: "拉長臉型，增加垂直線條。" },
    { code: "FS02", name: "圓臉(骨)", feature: "臉長與臉寬接近，骨架較明顯。", direction: "增加垂直線條，避免兩側膨脹。" },
    { code: "FS03", name: "橢圓臉(柔)", feature: "比例均衡，下巴柔和。", direction: "維持比例平衡。" },
    { code: "FS04", name: "橢圓臉(骨)", feature: "比例均衡，骨架清晰。", direction: "維持骨架線條。" },
    { code: "FS05", name: "長臉(柔)", feature: "臉長較明顯，輪廓柔和。", direction: "增加橫向比例，避免過度拉長。" },
    { code: "FS06", name: "長臉(骨)", feature: "臉長明顯，骨架感較強。", direction: "增加橫向比例，柔化直長線條。" },
    { code: "FS07", name: "方臉(柔)", feature: "下顎寬但線條不銳利。", direction: "柔化下顎，增加曲線修飾。" },
    { code: "FS08", name: "方臉(骨)", feature: "下顎線明顯，骨感較強。", direction: "柔化骨架，避免過度直硬線條。" },
    { code: "FS09", name: "菱形臉(柔)", feature: "顴骨較明顯，輪廓偏柔。", direction: "修飾顴骨，平衡上下面部比例。" },
    { code: "FS10", name: "菱形臉(骨)", feature: "顴骨突出，骨架存在感強。", direction: "降低顴骨銳利感，增加柔和過渡。" },
    { code: "FS11", name: "心形臉", feature: "上庭較寬，下巴較尖。", direction: "平衡額頭寬度與下巴尖度。" },
    { code: "FS12", name: "倒三角臉", feature: "上寬下窄，視覺重心在上半臉。", direction: "增加下半臉穩定感。" },
    { code: "FS13", name: "三角臉", feature: "下半臉較寬，額頭較窄。", direction: "增加上半臉輕盈與平衡感。" },
    { code: "FS14", name: "長方臉", feature: "臉長與方感同時明顯。", direction: "縮短視覺長度並柔化下顎線。" },
  ],
  ratio: [
    { code: "RT01", name: "短中庭", direction: "增加垂直比例。" },
    { code: "RT02", name: "長中庭", direction: "縮短中庭感，調整眉眼與腮紅重心。" },
    { code: "RT03", name: "短上庭", direction: "增加上庭高度。" },
    { code: "RT04", name: "長上庭", direction: "降低上庭比例。" },
    { code: "RT05", name: "短下庭", direction: "增加下庭延伸感。" },
    { code: "RT06", name: "長下庭", direction: "縮短下庭比例。" },
    { code: "RT07", name: "比例平衡", direction: "維持比例。" },
  ],
  line: [
    { code: "LC01", name: "直線主導(強)", direction: "適合俐落、乾淨、結構感。" },
    { code: "LC02", name: "直線主導(中)", direction: "可保留線條感，避免過度圓潤。" },
    { code: "LC03", name: "直曲混合(偏直)", direction: "以乾淨線條為主，保留少量柔和。" },
    { code: "LC04", name: "直曲平衡", direction: "造型彈性高，可依風格調整。" },
    { code: "LC05", name: "直曲混合(偏曲)", direction: "可加入柔和曲線與空氣感。" },
    { code: "LC06", name: "曲線主導(中)", direction: "適合柔和、圓潤、自然過渡。" },
    { code: "LC07", name: "曲線主導(強)", direction: "曲線感明顯，避免過度硬線條。" },
  ],
  volume: [
    { code: "VM01", name: "小量感", direction: "避免過重造型，維持輕盈比例。" },
    { code: "VM02", name: "中量感", direction: "造型彈性佳，依風格調整濃淡。" },
    { code: "VM03", name: "中偏大量感", direction: "可承接較有存在感的髮型與配件。" },
    { code: "VM04", name: "大量感", direction: "可使用明確輪廓與較強存在感造型。" },
  ],
  style: [
    { code: "ST01", name: "Cute", direction: "保留可愛、圓潤、親和感。" },
    { code: "ST02", name: "Active Cute", direction: "增加活潑與輕快感。" },
    { code: "ST03", name: "Fresh", direction: "清爽、自然、輕盈。" },
    { code: "ST04", name: "Soft Elegant", direction: "柔和、優雅、低壓迫感。" },
    { code: "ST05", name: "Feminine", direction: "女性感、柔美、細緻。" },
    { code: "ST06", name: "Glam Elegant", direction: "華麗、成熟、有存在感。" },
    { code: "ST07", name: "Cool", direction: "俐落、冷感、結構清楚。" },
    { code: "ST08", name: "Cool Casual", direction: "輕鬆、俐落、個性自然。" },
  ],
  age: [
    { code: "AG01", name: "幼態", direction: "保留輕盈與親和感。" },
    { code: "AG02", name: "偏幼態", direction: "可微調成熟度，但不宜過重。" },
    { code: "AG03", name: "年輕自然", direction: "維持自然清爽感。" },
    { code: "AG04", name: "成熟平衡", direction: "可走穩定、自然、質感路線。" },
    { code: "AG05", name: "成熟", direction: "可增加精緻感與結構。" },
    { code: "AG06", name: "成熟偏強", direction: "避免過度厚重，需保留氣色與柔和度。" },
  ],
};

const sheetConfig = {
  face: { sheet: SHEETS.face, codeKeys: ["FaceShapeCode", "臉型Code", "FaceShapeID", "code", "Code"], nameKeys: ["臉型名稱", "名稱", "FaceShapeName", "name", "Name"] },
  ratio: { sheet: SHEETS.ratio, codeKeys: ["RatioCode", "比例Code", "RTCode", "code", "Code"], nameKeys: ["比例名稱", "名稱", "RatioName", "name", "Name"] },
  line: { sheet: SHEETS.line, codeKeys: ["LineCode", "直曲Code", "LCCode", "code", "Code"], nameKeys: ["直曲名稱", "直曲類型", "名稱", "LineName", "name", "Name"] },
  volume: { sheet: SHEETS.volume, codeKeys: ["VolumeCode", "量感Code", "VMCode", "code", "Code"], nameKeys: ["量感名稱", "量感類型", "名稱", "VolumeName", "name", "Name"] },
  style: { sheet: SHEETS.style, codeKeys: ["StyleCode", "風格Code", "STCode", "code", "Code"], nameKeys: ["風格名稱", "名稱", "StyleName", "name", "Name"] },
  age: { sheet: SHEETS.age, codeKeys: ["AgeCode", "年齡Code", "AGCode", "code", "Code"], nameKeys: ["年齡名稱", "年齡感名稱", "年齡感類型", "名稱", "AgeName", "name", "Name"] },
};

const aiCodeMaps = {
  face: {
    圓臉骨: "FS02", 圓臉: "FS01", 圓形臉: "FS01",
    橢圓臉骨: "FS04", 橢圓臉: "FS03", 鵝蛋臉: "FS03",
    長臉骨: "FS06", 長臉: "FS05", 長形臉: "FS05",
    方臉骨: "FS08", 方臉: "FS07", 方形臉: "FS07",
    菱形臉骨: "FS10", 菱形臉: "FS09", 心形臉: "FS11",
    倒三角臉: "FS12", 三角臉: "FS13", 長方臉: "FS14",
  },
  ratio: { 短中庭: "RT01", 長中庭: "RT02", 短上庭: "RT03", 長上庭: "RT04", 短下庭: "RT05", 長下庭: "RT06", 比例平衡: "RT07", 平衡: "RT07" },
  line: { 直線主導強: "LC01", 直線主導中: "LC02", 直線偏多: "LC02", 直曲混合偏直: "LC03", 偏直: "LC03", 直曲平衡: "LC04", 平衡: "LC04", 直曲混合偏曲: "LC05", 偏曲: "LC05", 曲線主導中: "LC06", 曲線偏多: "LC06", 曲線主導強: "LC07" },
};

const colorSeasonGroups = {
  Spring: ["Light Spring", "Clear Spring", "Strong Spring"],
  Summer: ["Light Summer", "Clear Summer", "Mute Summer"],
  Autumn: ["Mute Autumn", "Deep Autumn", "Dark Autumn"],
  Winter: ["Strong Winter", "Deep Winter", "Dark Winter"],
};

// 風格座標圖：請放公開可讀的圖片網址，例如 Google Drive 圖片公開連結或圖床網址。
const STYLE_MAP_IMAGE_URL = "https://drive.google.com/thumbnail?id=1qJ-qTIeGXjYeh3IFW8qecjQ79GP1POIk&sz=w1600";

const ageAssessmentItems = [
  { key: "faceRatio", label: "面部比例", left: "圓臉／少於 1.4", right: "長臉／大於 1.4" },
  { key: "thirds", label: "三庭比例", left: "上庭較長", right: "下庭較長" },
  { key: "featureSize", label: "五官大小", left: "整體偏小", right: "整體偏大" },
  { key: "eyeDistance", label: "眼間距", left: "較開", right: "較近" },
  { key: "dimension", label: "面部立體感", left: "正面留白多；側面扁平", right: "五官佔比大；側面立體" },
];

const lineAssessmentItems = [
  { key: "outline", label: "面部外輪廓", left: "骨感有稜角", right: "飽滿流暢" },
  { key: "eyes", label: "眼睛", left: "細長", right: "圓眼" },
  { key: "lips", label: "嘴唇", left: "整體偏小", right: "厚" },
  { key: "nose", label: "鼻", left: "鼻樑明顯", right: "圓潤" },
  { key: "brows", label: "眉毛", left: "濃密、直", right: "淺淡、彎" },
];

function countAssessment(value = {}, leftValue, rightValue) {
  const vals = Object.values(value || {});
  return {
    left: vals.filter((item) => item === leftValue).length,
    right: vals.filter((item) => item === rightValue).length,
  };
}

function getAgeAssessmentResult(value = {}) {
  const counts = countAssessment(value, "juvenile", "mature");
  if (counts.left >= 3 && counts.left > counts.right) return `偏幼態可愛（幼態 ${counts.left}／成熟 ${counts.right}）`;
  if (counts.right >= 3 && counts.right > counts.left) return `偏成熟穩重（幼態 ${counts.left}／成熟 ${counts.right}）`;
  if (counts.left || counts.right) return `年齡感平衡（幼態 ${counts.left}／成熟 ${counts.right}）`;
  return "尚未評估";
}

function getLineAssessmentResult(value = {}) {
  const counts = countAssessment(value, "straight", "curve");
  if (counts.left >= 3 && counts.left > counts.right) return `直線感較多（直線 ${counts.left}／曲線 ${counts.right}）`;
  if (counts.right >= 3 && counts.right > counts.left) return `曲線感較多（直線 ${counts.left}／曲線 ${counts.right}）`;
  if (counts.left || counts.right) return `直曲比例平衡（直線 ${counts.left}／曲線 ${counts.right}）`;
  return "尚未評估";
}

function buildAgeAssessmentText(value = {}) {
  const lines = ageAssessmentItems.map((item) => {
    const selected = value[item.key];
    if (!selected) return "";
    return `${item.label}=${selected === "juvenile" ? item.left : item.right}`;
  }).filter(Boolean);
  return lines.join("；");
}

function buildLineAssessmentText(value = {}) {
  const lines = lineAssessmentItems.map((item) => {
    const selected = value[item.key];
    if (!selected) return "";
    return `${item.label}=${selected === "straight" ? item.left : item.right}`;
  }).filter(Boolean);
  return lines.join("；");
}

const lineVisualDefinitions = {
  LC01: "骨架感與線條感非常明顯，輪廓俐落、五官偏銳利或直線感強。整體視覺感偏冷、乾淨、有距離感，適合清楚結構與俐落輪廓，避免過度圓潤、甜美或鬆散的造型。",
  LC02: "骨架與輪廓清晰，但銳利感沒有 LC01 那麼強。整體視覺感偏俐落、乾淨、理性，可保留線條感，同時加入少量柔和細節，避免過硬或過度尖銳。",
  LC03: "輪廓中有直線架構，但仍帶一點柔和曲線。整體視覺感偏乾淨、有精神，但不至於冷硬。適合以直線為主、曲線為輔，保留清爽結構與適度親和感。",
  LC04: "臉部直曲比例均衡，既不過度銳利，也不過度圓潤。整體視覺感協調、彈性高，可依風格在清爽、柔和、優雅或俐落之間調整，是最容易跨風格微調的類型。",
  LC05: "臉部輪廓柔和，但仍保有一定結構。整體視覺感偏柔、親和、舒服，不會太幼或太甜。適合柔和曲線與乾淨線條並用，避免過度冷硬或過度銳利。",
  LC06: "輪廓圓潤、線條柔和，曲線感明顯。整體視覺感偏親和、柔軟、有溫度，適合自然圓潤、柔和過渡的造型，避免太直、太硬、太銳利的線條。",
  LC07: "臉部曲線感非常強，輪廓與五官多半柔和圓潤。整體視覺感非常柔軟、甜美或親和，適合保留圓潤與柔和感，但需用適度結構避免顯得過於稚嫩或鬆散。",
};

const ageVisualDefinitions = {
  AG01: "臉部比例偏短，五官多半圓潤、柔和、可愛感高。整體視覺感偏年輕、柔軟、親和，給人無距離感、容易親近的印象。若造型過成熟或線條過銳利，容易產生違和感。",
  AG02: "五官柔和、氣質年輕，整體仍有輕盈、親近、柔軟的視覺感，但比幼態更自然，不一定是可愛感強。適合保留清爽與親和，同時微微增加精緻度，避免造型太厚重。",
  AG03: "五官比例自然，氣質清爽、乾淨，整體視覺感年輕但不過度可愛。給人的感覺是自然、舒服、沒有壓迫感，適合走清新、簡潔、日常有質感的方向。造型重點是維持乾淨感，不需要過度堆疊成熟元素。",
  AG04: "五官比例成熟但仍保有柔和感，整體視覺感穩定、自然、有親和力。不是幼態，也不會過度強勢，適合走有質感、有整理感、但不壓迫的形象路線。造型上可增加精緻度與結構感，但仍需保留柔和與自然。",
  AG05: "骨架穩定，五官或氣質帶有明顯成熟感，整體視覺感較精緻、穩重、有女人味或專業感。適合使用較完整的造型結構來提升質感，但要避免過度老氣或太厚重，重點是成熟而不沉重。",
  AG06: "氣場穩定、存在感高，五官、骨架或整體氣質容易給人較強烈、成熟、有距離感的印象。適合乾淨、有結構、有品質感的造型，但需控制厚重度與銳利感，避免讓氣場過強、壓迫或顯老。",
};

const faceDetailTagSections = [
  {
    title: "臉型偏向",
    hint: "主臉型已經選定，這裡補充偏向的架構，可複選。",
    options: ["偏長", "偏方", "偏菱形", "偏圓"],
  },
  {
    title: "下顎／下巴",
    hint: "補充下半臉線條與下巴狀態。",
    options: ["下顎略方", "下巴偏窄", "下巴偏短", "下巴偏尖", "下巴戽斗"],
  },
  {
    title: "顴骨",
    hint: "補充顴骨存在感與外擴狀態。",
    options: ["顴骨略明顯", "顴骨外擴"],
  },
  {
    title: "太陽穴／兩頰",
    hint: "補充臉側凹陷與支撐感。",
    options: ["太陽穴凹陷", "兩頰凹陷"],
  },
];

const observationTagGroups = {
  faceDetailTags: {
    title: "臉型細節標籤",
    hint: "主臉型已選定，這裡只補充偏向與局部細節。",
    options: faceDetailTagSections.flatMap((section) => section.options),
  },
  ratioFocusTags: {
    title: "比例與重心標籤",
    hint: "三庭長度已可自動判讀比例，這裡只補充視覺重心感受。",
    options: ["視覺重心偏上", "視覺重心平衡", "視覺重心偏下"],
  },
  featureStructureTags: {
    title: "五官結構標籤",
    hint: "不放眼距，眼距請填在「五眼／眼距觀察」。",
    options: ["五官量感偏小但需要支撐", "五官存在感強", "眉峰偏高", "眉骨量感低", "眼下凹陷", "眼袋感明顯", "面中立體度不足", "鼻樑存在感低", "嘴唇量感偏小", "嘴唇量感明顯"],
  },
  correctionGoalTags: {
    title: "修飾目標標籤",
    hint: "這裡只選結構修飾目標，不選風格限制。",
    options: ["降低橫向寬度", "增加縱向延伸", "修飾下顎方感", "修飾顴骨存在感", "修飾太陽穴凹陷", "修飾兩頰凹陷", "提升五官集中度", "建立中段聚焦", "拉提視覺重心", "增加結構支撐", "柔化骨架銳利感", "保留臉部柔和感"],
  },
  styleSupplementTags: {
    title: "副風格",
    hint: "主風格已由 ST 決定，這裡只選一個副風格；沒有副風格可不選。",
    options: ["Cute", "Active Cute", "Fresh", "Soft Elegant", "Feminine", "Glam Elegant", "Cool", "Cool Casual"],
  },
};

function cx(...classes) {
  return classes.filter(Boolean).join(" ");
}

function normalizeText(text) {
  return String(text || "").replace(/[\s｜|／/（）()【】「」『』：:、，,。\.\-＿_]/g, "").trim();
}

function mapAIResultToCodes(result) {
  const faceText = normalizeText(result?.faceShape ?? result?.face ?? "");
  const ratioText = normalizeText(result?.ratio ?? "");
  const lineText = normalizeText(result?.line ?? "");
  const directFaceCode = String(result?.faceCode ?? "").trim();
  const directRatioCode = String(result?.ratioCode ?? "").trim();
  const directLineCode = String(result?.lineCode ?? "").trim();
  const directVolumeCode = String(result?.volumeCode ?? "").trim();
  const directStyleCode = String(result?.styleCode ?? "").trim();
  const directAgeCode = String(result?.ageCode ?? "").trim();
  const findCode = (text, map) => {
    const key = Object.keys(map).sort((a, b) => normalizeText(b).length - normalizeText(a).length).find((k) => text.includes(normalizeText(k)));
    return key ? map[key] : "";
  };
  return {
    face: directFaceCode || findCode(faceText, aiCodeMaps.face),
    ratio: directRatioCode || findCode(ratioText, aiCodeMaps.ratio),
    line: directLineCode || findCode(lineText, aiCodeMaps.line),
    volume: directVolumeCode,
    style: directStyleCode,
    age: directAgeCode,
    confidence: Number(result?.confidence ?? 0),
    reason: String(result?.reason ?? result?.summary ?? ""),
  };
}

function firstValue(item, keys) {
  for (const key of keys) {
    const value = item?.[key];
    if (value !== undefined && value !== null && String(value).trim() !== "") return String(value).trim();
  }
  return "";
}

function normalizeRows(rawRows, type) {
  if (!Array.isArray(rawRows)) return [];
  const config = sheetConfig[type];
  if (!config) return [];
  return rawRows
    .map((item) => {
      const code = firstValue(item, config.codeKeys);
      const name = firstValue(item, config.nameKeys);
      if (!code || !name) return null;
      return {
        code,
        name,
        feature: String(item["視覺特徵／視覺感"] ?? item["視覺特徵/視覺感"] ?? item["視覺感"] ?? item["視覺說明"] ?? item["臉型特徵"] ?? item["比例特徵"] ?? item["視覺特徵"] ?? item["風格特徵"] ?? item["量感特徵"] ?? item["年齡感特徵"] ?? item["特徵"] ?? item.feature ?? "").trim(),
        direction: String(item["修飾方向"] ?? item["修飾策略"] ?? item["修飾效果"] ?? item["適合類型"] ?? item["說明"] ?? item.direction ?? "").trim(),
        hairDirection: String(item["建議髮型方向"] ?? item["髮型建議"] ?? item["髮型策略"] ?? item["適合髮型"] ?? "").trim(),
        avoidHairDirection: String(item["避免髮型"] ?? item["不適合髮型"] ?? item["避免方向"] ?? "").trim(),
        bangDirection: String(item["建議瀏海方向"] ?? item["瀏海建議"] ?? item["瀏海策略"] ?? item["適合瀏海"] ?? "").trim(),
        partDirection: String(item["建議分線方向"] ?? item["分線建議"] ?? item["分線策略"] ?? "").trim(),
        glassesDirection: String(item["建議眼鏡方向"] ?? item["眼鏡建議"] ?? item["眼鏡策略"] ?? item["適合眼鏡"] ?? "").trim(),
        earringsDirection: String(item["建議耳環方向"] ?? item["耳環建議"] ?? item["耳環策略"] ?? item["適合耳環"] ?? "").trim(),
        makeupDirection: String(item["建議妝容方向"] ?? item["妝容重點"] ?? item["妝容方向"] ?? item["妝容建議"] ?? item["妝容策略"] ?? item["適合妝容"] ?? "").trim(),
        avoidDirection: String(item["避免造型"] ?? item["避免策略"] ?? item["避免類型"] ?? item["避免建議"] ?? "").trim(),
        status: String(item.Status ?? item["狀態"] ?? "Active").trim().toLowerCase(),
      };
    })
    .filter(Boolean)
    .filter((item) => !item.status || item.status === "active");
}

function getName(options, code) {
  return options.find((item) => item.code === code)?.name || "未選擇";
}

function getRecord(options, code) {
  return options.find((item) => item.code === code) || null;
}

function toNumber(value) {
  const num = Number(String(value || "").replace(/[a-zA-Z\s]/g, ""));
  return Number.isFinite(num) && num > 0 ? num : 0;
}

function calculateFaceMeasurements(measurements) {
  const faceLength = toNumber(measurements.faceLength);
  const faceWidth = toNumber(measurements.faceWidth);
  const upper = toNumber(measurements.upperThird);
  const middle = toNumber(measurements.middleThird);
  const lower = toNumber(measurements.lowerThird);
  const lengthWidthRatio = faceLength && faceWidth ? faceLength / faceWidth : 0;
  const thirdsTotal = upper + middle + lower;
  const upperPct = thirdsTotal ? (upper / thirdsTotal) * 100 : 0;
  const middlePct = thirdsTotal ? (middle / thirdsTotal) * 100 : 0;
  const lowerPct = thirdsTotal ? (lower / thirdsTotal) * 100 : 0;
  let faceRatioLabel = "尚未輸入完整長寬";
  if (lengthWidthRatio) {
    if (lengthWidthRatio < 1.18) faceRatioLabel = "偏短寬 / 圓感較明顯";
    else if (lengthWidthRatio <= 1.35) faceRatioLabel = "長寬比例平衡";
    else faceRatioLabel = "偏長臉 / 縱向感較明顯";
  }
  let thirdsLabel = "尚未輸入完整三庭";
  if (upper && middle && lower) {
    const values = [upper, middle, lower];
    const gap = Math.max(...values) - Math.min(...values);
    const max = Math.max(...values);
    if (gap <= Math.max(thirdsTotal * 0.06, 0.1)) thirdsLabel = "三庭比例接近平衡";
    else if (max === upper) thirdsLabel = "上庭比例偏長";
    else if (max === middle) thirdsLabel = "中庭比例偏長";
    else thirdsLabel = "下庭比例偏長";
  }
  return { lengthWidthRatio, upperPct, middlePct, lowerPct, faceRatioLabel, thirdsLabel };
}

async function fetchSheetRows(sheetName) {
  const url = `${API_BASE_URL}?sheet=${encodeURIComponent(sheetName)}`;
  const response = await fetch(url, { method: "GET" });
  const text = await response.text();
  if (!response.ok) throw new Error(`HTTP ${response.status}: ${text.slice(0, 160)}`);
  let parsed;
  try {
    parsed = JSON.parse(text);
  } catch {
    throw new Error(`API 沒有回傳 JSON，前 160 字：${text.slice(0, 160)}`);
  }
  if (parsed?.error) throw new Error(String(parsed.error));
  if (!Array.isArray(parsed)) throw new Error("API 回傳格式不是陣列。");
  return parsed;
}

async function requestAIInitialAnalysis(photoBase64, provider = "gemini") {
  const response = await fetch(API_BASE_URL, {
    method: "POST",
    body: JSON.stringify({
      action: "aiAnalyzeFace",
      imageBase64: photoBase64,
      provider,
    }),
  });

  const result = await response.json();
  if (!result.ok) throw new Error(result.error || "AI 初判臉型／比例／直曲失敗");
  return result.result;
}

async function requestAIProportionAnalysis(photoBase64, provider = "gemini") {
  const response = await fetch(API_BASE_URL, {
    method: "POST",
    body: JSON.stringify({
      action: "aiAnalyzeProportion",
      imageBase64: photoBase64,
      provider,
    }),
  });

  const result = await response.json();
  if (!result.ok) throw new Error(result.error || "AI 初判三庭五眼失敗");
  return result.result;
}

function formatClientError(error) {
  const message = error instanceof Error ? error.message : String(error || "");
  if (message.includes("Quota exceeded") || message.includes("exceeded your current quota") || message.includes("free_tier_requests")) {
    return "Gemini API 免費額度或短時間請求次數已達上限，請稍後再試，或到 Google AI Studio / Google Cloud 調整用量與計費設定。";
  }
  if (message.includes("high demand")) {
    return "Gemini 模型目前使用量較高，請稍後再試一次。";
  }
  return message;
}

function Pill({ children, active, onClick }) {
  return (
    <button type="button" onClick={onClick} className={cx("rounded-full border px-4 py-2 text-sm transition", active ? "border-stone-900 bg-stone-900 text-white shadow-sm" : "border-stone-200 bg-white text-stone-600 hover:border-stone-400 hover:bg-stone-50")}>
      {children}
    </button>
  );
}

function Field({ label, hint, children }) {
  return (
    <div className="space-y-2">
      <div>
        <div className="text-sm font-medium text-stone-700">{label}</div>
        {hint && <div className="mt-1 text-xs text-stone-400">{hint}</div>}
      </div>
      {children}
    </div>
  );
}

function SingleChoiceGroup({ label, value, options, onChange }) {
  return (
    <div className="space-y-2">
      <div className="text-sm font-medium text-stone-700">{label}</div>
      <div className="flex flex-wrap gap-2">
        {options.map((item) => {
          const active = value === item;
          return (
            <button
              key={item}
              type="button"
              onClick={() => onChange(active ? "" : item)}
              className={cx(
                "rounded-full border px-4 py-2 text-sm transition",
                active
                  ? "border-rose-700 bg-rose-600 text-white shadow-sm"
                  : "border-rose-200 bg-rose-50 text-stone-700 hover:border-rose-400 hover:bg-rose-100"
              )}
            >
              {active ? "✓ " : ""}{item}
            </button>
          );
        })}
      </div>
    </div>
  );
}

function OptionGrid({ options, value, onChange, loading }) {
  if (loading) return <div className="rounded-3xl border border-stone-200 bg-stone-50 p-5 text-sm text-stone-500">正在讀取試算表資料…</div>;
  return (
    <div className="flex flex-wrap gap-2">
      {options.map((item) => (
        <Pill key={item.code} active={value === item.code} onClick={() => onChange(item.code)}>
          <span className="mr-1 text-xs opacity-60">{item.code}</span>{item.name}
        </Pill>
      ))}
    </div>
  );
}

const faceShapeUiCopy = {
  FS01: "臉長≈臉寬，輪廓圓潤，脂肪感高。",
  FS02: "臉長≈臉寬，顴骨略明顯，骨架感較強。",
  FS03: "比例均衡，下巴柔和。",
  FS04: "比例均衡，骨架清晰，顴骨略明顯。",
  FS05: "臉長明顯，輪廓柔和。",
  FS06: "臉長比例高，骨架感強，下巴略方。",
  FS07: "下顎寬，但線條柔和。",
  FS08: "下顎角明顯，骨架強。",
  FS09: "顴骨最寬，額頭窄，下巴尖。",
  FS10: "顴骨突出，骨架感強。",
  FS11: "額頭寬，下巴尖。",
  FS12: "額頭寬，下巴尖銳。",
  FS13: "額頭窄，下顎寬。",
  FS14: "臉長且下顎直，骨架明顯。",
};

// 圖片版臉型卡片：
// 若妳之後把 14 張小圖上傳到 public/face-shapes/，只要把網址填在這裡，前端就會優先顯示圖片。
// 若某一張沒有網址，系統會自動退回使用 MiniFaceLineArt 線稿版，不會壞掉。
const faceShapeImageMap = {
  FS01: "",
  FS02: "",
  FS03: "",
  FS04: "",
  FS05: "",
  FS06: "",
  FS07: "",
  FS08: "",
  FS09: "",
  FS10: "",
  FS11: "",
  FS12: "",
  FS13: "",
  FS14: "",
};

function MiniFaceLineArt({ code, active }) {
  const ink = active ? "#ffffff" : "#6b4f45";
  const accent = active ? "#fff1f2" : "#be7082";
  const blush = active ? "#ffffff" : "#f5c7cf";
  const guide = active ? "#ffffff" : "#d58a9a";

  const outlines = {
    FS01: "M70 37 C93 38 108 58 108 83 C108 111 92 132 70 132 C48 132 32 111 32 83 C32 58 47 38 70 37 Z",
    FS02: "M70 35 C92 36 106 56 106 82 C106 108 92 130 70 132 C48 130 34 108 34 82 C34 56 48 36 70 35 Z",
    FS03: "M70 30 C90 31 103 55 103 83 C103 113 88 136 70 136 C52 136 37 113 37 83 C37 55 50 31 70 30 Z",
    FS04: "M70 29 C91 31 103 55 104 82 C105 111 88 136 70 137 C52 136 35 111 36 82 C37 55 49 31 70 29 Z",
    FS05: "M70 22 C89 23 101 55 101 89 C101 124 88 146 70 146 C52 146 39 124 39 89 C39 55 51 23 70 22 Z",
    FS06: "M70 22 C91 24 103 55 103 89 L99 125 C94 139 84 147 70 147 C56 147 46 139 41 125 L37 89 C37 55 49 24 70 22 Z",
    FS07: "M70 34 C91 34 105 56 105 83 L101 117 C94 130 83 136 70 136 C57 136 46 130 39 117 L35 83 C35 56 49 34 70 34 Z",
    FS08: "M70 33 C92 33 107 55 107 83 L103 120 C96 134 84 139 70 139 C56 139 44 134 37 120 L33 83 C33 55 48 33 70 33 Z",
    FS09: "M70 28 C88 33 102 58 112 84 C101 113 86 138 70 146 C54 138 39 113 28 84 C38 58 52 33 70 28 Z",
    FS10: "M70 27 C89 32 104 57 115 84 C103 116 87 141 70 149 C53 141 37 116 25 84 C36 57 51 32 70 27 Z",
    FS11: "M70 145 C47 119 32 90 34 57 C36 34 54 28 70 45 C86 28 104 34 106 57 C108 90 93 119 70 145 Z",
    FS12: "M70 148 C50 123 36 88 30 35 L110 35 C104 88 90 123 70 148 Z",
    FS13: "M70 28 C88 55 106 94 113 142 L27 142 C34 94 52 55 70 28 Z",
    FS14: "M70 18 C91 19 104 49 104 86 L101 132 C95 146 84 151 70 151 C56 151 45 146 39 132 L36 86 C36 49 49 19 70 18 Z",
  };

  const isBone = ["FS02", "FS04", "FS06", "FS08", "FS10", "FS14"].includes(code);
  const cheekCodes = ["FS02", "FS04", "FS09", "FS10"].includes(code);
  const jawCodes = ["FS06", "FS07", "FS08", "FS13", "FS14"].includes(code);
  const narrowTopCodes = ["FS09", "FS10", "FS13"].includes(code);
  const wideTopCodes = ["FS11", "FS12"].includes(code);

  return (
    <svg viewBox="0 0 140 168" className="h-40 w-full" aria-hidden="true">
      <defs>
        <filter id={`mini-face-shadow-${code}`} x="-20%" y="-20%" width="140%" height="140%">
          <feDropShadow dx="0" dy="10" stdDeviation="8" floodColor="#9f1239" floodOpacity={active ? "0.18" : "0.08"} />
        </filter>
      </defs>

      <rect x="27" y="15" width="86" height="140" rx="40" fill={active ? "rgba(255,255,255,0.12)" : "#fff7f7"} opacity="0.9" />

      <path
        d={outlines[code] || outlines.FS03}
        fill="none"
        stroke={accent}
        strokeWidth={isBone ? 3.2 : 2.4}
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeDasharray={isBone ? "0" : "6 5"}
        filter={`url(#mini-face-shadow-${code})`}
      />

      <path d="M42 42 C49 23 91 23 98 42" fill="none" stroke={ink} strokeWidth="1.8" strokeLinecap="round" opacity="0.45" />
      <path d="M46 36 C55 17 85 17 96 36" fill="none" stroke={ink} strokeWidth="1" strokeLinecap="round" opacity="0.25" />

      <line x1="70" y1="36" x2="70" y2="137" stroke={guide} strokeWidth="1.2" opacity="0.7" strokeDasharray="5 5" />
      <line x1="43" y1="82" x2="97" y2="82" stroke={guide} strokeWidth="1.2" opacity="0.55" strokeDasharray="5 5" />

      {narrowTopCodes && <line x1="50" y1="48" x2="90" y2="48" stroke={guide} strokeWidth="1.7" strokeLinecap="round" opacity="0.8" />}
      {wideTopCodes && <line x1="34" y1="48" x2="106" y2="48" stroke={guide} strokeWidth="1.7" strokeLinecap="round" opacity="0.8" />}
      {jawCodes && <line x1="40" y1="119" x2="100" y2="119" stroke={guide} strokeWidth="1.7" strokeLinecap="round" opacity="0.8" />}
      {cheekCodes && (
        <>
          <circle cx="39" cy="83" r="3" fill={accent} opacity="0.78" />
          <circle cx="101" cy="83" r="3" fill={accent} opacity="0.78" />
        </>
      )}

      <path d="M49 73 C55 69 60 69 64 73" fill="none" stroke={ink} strokeWidth="2" strokeLinecap="round" />
      <path d="M76 73 C80 69 85 69 91 73" fill="none" stroke={ink} strokeWidth="2" strokeLinecap="round" />
      <circle cx="56" cy="76" r="2.1" fill={ink} opacity="0.85" />
      <circle cx="84" cy="76" r="2.1" fill={ink} opacity="0.85" />
      <path d="M70 79 C67 87 67 93 70 96 C73 93 73 87 70 79 Z" fill="none" stroke={ink} strokeWidth="1.2" opacity="0.55" />
      <path d="M58 112 Q70 117 82 112" fill="none" stroke={ink} strokeWidth="1.8" strokeLinecap="round" opacity="0.85" />

      <ellipse cx="45" cy="93" rx="6" ry="4" fill={blush} opacity={active ? "0.28" : "0.55"} />
      <ellipse cx="95" cy="93" rx="6" ry="4" fill={blush} opacity={active ? "0.28" : "0.55"} />
    </svg>
  );
}

function FaceShapeIcon({ code, active }) {
  const imageUrl = faceShapeImageMap[code];
  if (imageUrl) {
    return (
      <div className="flex h-40 w-full items-center justify-center overflow-hidden rounded-3xl bg-white">
        <img src={imageUrl} alt={`${code} 臉型示意圖`} className="h-full w-full object-contain" />
      </div>
    );
  }

  return <MiniFaceLineArt code={code} active={active} />;
}

function FaceShapeGrid({ options, value, onChange, loading }) {
  if (loading) {
    return (
      <div className="rounded-3xl border border-stone-200 bg-stone-50 p-5 text-sm text-stone-500">
        正在讀取臉型資料…
      </div>
    );
  }

  return (
    <div className="rounded-[2rem] border border-rose-100 bg-white p-4 shadow-sm sm:p-5">
      <div className="mb-5 flex flex-col gap-1">
        <div className="text-sm font-semibold text-stone-900">臉型圖卡選擇</div>
        <div className="text-xs leading-5 text-stone-500">選擇最符合輪廓特徵的臉型；FS 代碼僅保留為系統對應，不作為主要視覺。</div>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-5">
        {options.map((item, index) => {
          const active = value === item.code;
          return (
            <button
              key={item.code}
              type="button"
              onClick={() => onChange(item.code)}
              className={cx(
                "relative overflow-hidden rounded-[1.5rem] border bg-white px-4 pb-4 pt-3 text-center transition-all duration-200",
                active
                  ? "border-rose-500 bg-rose-50 shadow-xl shadow-rose-100 ring-2 ring-rose-300"
                  : "border-rose-100 hover:-translate-y-0.5 hover:border-rose-300 hover:shadow-lg hover:shadow-stone-100"
              )}
            >
              <div className="absolute left-3 top-3 flex h-8 w-8 items-center justify-center rounded-full bg-rose-100 text-sm font-semibold text-rose-600">
                {index + 1}
              </div>

              {active && (
                <div className="absolute right-3 top-3 flex h-8 w-8 items-center justify-center rounded-full bg-rose-600 text-sm font-semibold text-white shadow-md">
                  ✓
                </div>
              )}

              <div className="pointer-events-none mt-4 flex justify-center">
                <div className="w-full max-w-[150px]">
                  <FaceShapeIcon code={item.code} active={false} />
                </div>
              </div>

              <div className="mt-1 text-lg font-semibold tracking-tight text-stone-900">{item.name}</div>
              <div className="mt-2 min-h-[2.5rem] text-xs leading-5 text-stone-500">
                {faceShapeUiCopy[item.code] || item.feature || "臉型特徵會依試算表資料帶入。"}
              </div>
              <div className="mt-3 text-[10px] font-semibold tracking-[0.18em] text-rose-300">{item.code}</div>
            </button>
          );
        })}
      </div>
    </div>
  );
}

function CheckboxTagGroup({ title, hint, options, value = [], onChange }) {
  const selected = Array.isArray(value) ? value : [];
  const toggle = (tag) => {
    const next = selected.includes(tag) ? selected.filter((item) => item !== tag) : [...selected, tag];
    onChange(next);
  };
  return (
    <div className="rounded-3xl border border-stone-200 bg-white p-5 shadow-sm">
      <div className="mb-4">
        <div className="text-sm font-semibold text-stone-900">{title}</div>
        {hint && <div className="mt-1 text-xs leading-5 text-stone-500">{hint}</div>}
      </div>
      <div className="flex flex-wrap gap-2">
        {options.map((tag) => {
          const active = selected.includes(tag);
          return (
            <button key={tag} type="button" onClick={() => toggle(tag)} className={cx("rounded-full border px-4 py-2 text-sm transition", active ? "border-rose-700 bg-rose-600 text-white shadow-sm" : "border-rose-200 bg-rose-50 text-stone-700 hover:border-rose-400 hover:bg-rose-100")}>
              {active ? "✓ " : ""}{tag}
            </button>
          );
        })}
      </div>
    </div>
  );
}

function GroupedCheckboxTagGroup({ title, hint, sections, value = [], onChange }) {
  const selected = Array.isArray(value) ? value : [];
  const toggle = (tag) => {
    const next = selected.includes(tag) ? selected.filter((item) => item !== tag) : [...selected, tag];
    onChange(next);
  };

  return (
    <div className="rounded-3xl border border-stone-200 bg-white p-5 shadow-sm">
      <div className="mb-5">
        <div className="text-sm font-semibold text-stone-900">{title}</div>
        {hint && <div className="mt-1 text-xs leading-5 text-stone-500">{hint}</div>}
      </div>
      <div className="grid gap-4 md:grid-cols-2">
        {sections.map((section) => (
          <div key={section.title} className="rounded-3xl border border-stone-100 bg-stone-50/60 p-4">
            <div className="mb-3">
              <div className="text-sm font-semibold text-stone-900">{section.title}</div>
              {section.hint && <div className="mt-1 text-xs leading-5 text-stone-500">{section.hint}</div>}
            </div>
            <div className="flex flex-wrap gap-2">
              {section.options.map((tag) => {
                const active = selected.includes(tag);
                return (
                  <button
                    key={tag}
                    type="button"
                    onClick={() => toggle(tag)}
                    className={cx(
                      "rounded-full border px-4 py-2 text-sm transition",
                      active
                        ? "border-rose-700 bg-rose-600 text-white shadow-sm"
                        : "border-rose-200 bg-rose-50 text-stone-700 hover:border-rose-400 hover:bg-rose-100"
                    )}
                  >
                    {active ? "✓ " : ""}{tag}
                  </button>
                );
              })}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function BinaryAssessmentTable({ title, hint, items, leftTitle, rightTitle, leftValue, rightValue, value = {}, onChange, result }) {
  const update = (key, nextValue) => {
    onChange({ ...(value || {}), [key]: value?.[key] === nextValue ? "" : nextValue });
  };

  return (
    <div className="rounded-3xl border border-stone-200 bg-white p-5 shadow-sm">
      <div className="mb-5">
        <div className="text-sm font-semibold text-stone-900">{title}</div>
        {hint && <div className="mt-1 text-xs leading-5 text-stone-500">{hint}</div>}
      </div>
      <div className="overflow-hidden rounded-3xl border border-stone-100">
        <div className="grid grid-cols-[1.1fr_1fr_1fr] bg-stone-900 text-sm font-semibold text-white">
          <div className="px-4 py-3">評估項目</div>
          <div className="px-4 py-3 text-center">{leftTitle}</div>
          <div className="px-4 py-3 text-center">{rightTitle}</div>
        </div>
        {items.map((item, index) => (
          <div key={item.key} className={cx("grid grid-cols-[1.1fr_1fr_1fr] items-stretch border-t border-stone-100 text-sm", index % 2 === 0 ? "bg-white" : "bg-stone-50/70")}>
            <div className="flex items-center px-4 py-3 font-medium text-stone-800">{item.label}</div>
            <button type="button" onClick={() => update(item.key, leftValue)} className={cx("px-4 py-3 text-center transition", value?.[item.key] === leftValue ? "bg-rose-600 text-white" : "text-stone-600 hover:bg-rose-50")}>{item.left}</button>
            <button type="button" onClick={() => update(item.key, rightValue)} className={cx("px-4 py-3 text-center transition", value?.[item.key] === rightValue ? "bg-rose-600 text-white" : "text-stone-600 hover:bg-rose-50")}>{item.right}</button>
          </div>
        ))}
      </div>
      <div className="mt-4 rounded-2xl bg-rose-50/60 px-4 py-3 text-sm font-medium text-rose-800">定位結論：{result}</div>
    </div>
  );
}

function SingleSelectTagGroup({ title, groups, value, onChange }) {
  return (
    <div className="rounded-3xl border border-stone-200 bg-white p-5 shadow-sm">
      <div className="mb-4 text-sm font-semibold text-stone-900">{title}</div>
      <div className="space-y-4">
        {Object.entries(groups).map(([groupName, items]) => (
          <div key={groupName}>
            <div className="mb-2 text-xs font-semibold uppercase tracking-widest text-stone-400">{groupName}</div>
            <div className="flex flex-wrap gap-2">
              {items.map((item) => {
                const active = value === item;
                return (
                  <button key={item} type="button" onClick={() => onChange(active ? "" : item)} className={cx("rounded-full border px-4 py-2 text-sm transition", active ? "border-rose-700 bg-rose-600 text-white shadow-sm" : "border-rose-200 bg-rose-50 text-stone-700 hover:border-rose-400 hover:bg-rose-100")}>
                    {active ? "✓ " : ""}{item}
                  </button>
                );
              })}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function ProgressRail({ current, setCurrent, data, setData }) {
  return (
    <header className="mb-8 rounded-[2rem] bg-gradient-to-r from-stone-950 via-stone-900 to-stone-800 p-5 shadow-2xl shadow-stone-900/25">
      <div className="grid gap-6 xl:grid-cols-[260px_1fr_360px] xl:items-start">
        <div className="flex items-center gap-3">
          <div className="flex h-14 w-14 shrink-0 items-center justify-center rounded-2xl bg-rose-500 text-sm font-bold tracking-widest text-white shadow-lg shadow-rose-500/40">
            JIS
          </div>
          <div>
            <div className="text-[10px] uppercase tracking-[0.18em] text-stone-500">Joanna Image System</div>
            <h1 className="text-xl font-semibold leading-tight text-white">顏分析顧問系統</h1>
          </div>
        </div>

        <div>
          <div className="mb-4 flex items-center justify-between gap-3">
            <div className="text-xs font-medium text-stone-500">分析步驟切換</div>
            <div className="rounded-full bg-stone-800 px-4 py-2 text-xs font-medium text-stone-400">
              Step {current} / {steps.length - 1} · {steps[current]?.label || "臉型 FS"}
            </div>
          </div>
          <div className="flex gap-2 overflow-x-auto pb-1">
            {steps.filter((step) => step.id !== "basic").map((step, index) => {
              const stepIndex = index + 1;
              const active = current === stepIndex;
              return (
                <button
                  key={step.id}
                  type="button"
                  onClick={() => setCurrent(stepIndex)}
                  className={cx(
                    "flex shrink-0 items-center gap-2 rounded-2xl px-4 py-2.5 text-sm transition-all duration-200",
                    active
                      ? "bg-rose-500 text-white shadow-lg shadow-rose-500/30"
                      : "bg-stone-800/70 text-stone-400 hover:bg-stone-700 hover:text-white"
                  )}
                >
                  <span className="flex h-7 w-7 items-center justify-center rounded-xl bg-white/10 text-xs">{step.icon}</span>
                  <span className="font-medium">{step.label}</span>
                </button>
              );
            })}
          </div>
        </div>

        <div className="rounded-[1.75rem] border border-white/10 bg-white/8 p-4 backdrop-blur">
          <div className="mb-3">
            <div className="text-xs uppercase tracking-[0.18em] text-white/35">Case Setup</div>
            <div className="mt-1 text-base font-semibold text-white">建立個案資料</div>
          </div>
          <div className="space-y-3">
            <div>
              <div className="mb-1 text-xs text-white/45">個案姓名 / 代號</div>
              <input
                value={data.clientName}
                onChange={(e) => setData((prev) => ({ ...prev, clientName: e.target.value }))}
                className="w-full rounded-2xl border border-white/10 bg-white/10 px-4 py-3 text-sm text-white outline-none placeholder:text-white/30 focus:border-rose-300"
                placeholder="例如：A小姐"
              />
            </div>
            <div>
              <div className="mb-1 text-xs text-white/45">主要需求</div>
              <select
                value={data.need}
                onChange={(e) => setData((prev) => ({ ...prev, need: e.target.value }))}
                className="w-full rounded-2xl border border-white/10 bg-white/10 px-4 py-3 text-sm text-white outline-none focus:border-rose-300"
              >
                <option className="text-stone-900">完整形象報告</option>
                <option className="text-stone-900">髮型建議</option>
                <option className="text-stone-900">妝容調整</option>
                <option className="text-stone-900">眼鏡 / 飾品</option>
                <option className="text-stone-900">形象照前建議</option>
              </select>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}

function ApiNotice({ apiState }) {
  const isSuccess = apiState.status === "success";
  return (
    <div className={cx("rounded-xl border px-4 py-3 text-sm font-medium", isSuccess ? "border-emerald-100 bg-emerald-50 text-emerald-700" : "border-amber-100 bg-amber-50 text-amber-700")}>
      {isSuccess ? "✓ 已連接試算表資料" : "○ 尚未讀取試算表，目前使用本機備用資料"}
      {apiState.error && <div className="mt-1 break-words text-xs opacity-80">錯誤：{apiState.error}</div>}
    </div>
  );
}

function StickyPhotoSummaryPanel({ data, setData, options, apiStates, saveState, onSaveCustomer, setCurrent }) {
  const fileInputRef = useRef(null);
  const connectedCount = Object.values(apiStates || {}).filter((state) => state.status === "success").length;

  const summaryRows = [
    ["臉型", getName(options.face, data.face), 1],
    ["比例", getName(options.ratio, data.ratio), 2],
    ["年齡感", getName(options.age, data.age), 3],
    ["直曲", getName(options.line, data.line), 4],
    ["量感", getName(options.volume, data.volume), 5],
    ["風格", getName(options.style, data.style), 6],
    ["色彩季型補充", data.colorSeason || "未選擇", 0],
  ];

  const handleFileChange = (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onloadend = () => setData((prev) => ({ ...prev, photo: reader.result }));
    reader.readAsDataURL(file);
  };

  return (
    <section className="mb-8 rounded-[2rem] border border-stone-200 bg-white/95 p-4 shadow-xl shadow-stone-200/70 backdrop-blur md:p-5">
      <div className="mb-4 flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
        <div>
          <div className="text-xs font-semibold uppercase tracking-[0.18em] text-stone-400">Consultant View</div>
          <h2 className="mt-1 text-lg font-semibold text-stone-900">固定觀察區</h2>
        </div>
        <div className="flex flex-wrap gap-2">
          <span className="rounded-full bg-stone-100 px-3 py-1.5 text-xs font-medium text-stone-600">{connectedCount}/6 已讀取</span>
          <button type="button" disabled={saveState?.status === "loading"} onClick={onSaveCustomer} className="rounded-full bg-stone-900 px-4 py-1.5 text-xs font-medium text-white disabled:cursor-not-allowed disabled:opacity-40">
            {saveState?.status === "loading" ? "儲存中…" : "儲存個案到 01"}
          </button>
        </div>
      </div>

      {saveState?.status === "success" && <div className="mb-4 rounded-2xl bg-emerald-50 px-4 py-2 text-xs leading-5 text-emerald-700">已儲存至第 {saveState.rowNumber} 列。</div>}
      {saveState?.error && <div className="mb-4 rounded-2xl bg-rose-50 px-4 py-2 text-xs leading-5 text-rose-700">{saveState.error}</div>}

      <div className="grid gap-5 xl:grid-cols-[320px_1fr]">
        <div>
          <div className="mb-3 flex items-center justify-between">
            <div>
              <div className="text-sm font-semibold text-stone-900">人像照片</div>
              <div className="mt-1 text-xs text-stone-400">分析過程中固定顯示</div>
            </div>
            <button type="button" onClick={() => fileInputRef.current?.click()} className="rounded-full border border-stone-200 bg-white px-4 py-2 text-xs text-stone-700 hover:bg-stone-50">
              {data.photo ? "更換照片" : "上傳照片"}
            </button>
          </div>

          <input ref={fileInputRef} type="file" className="hidden" accept="image/*" onChange={handleFileChange} />

          <button type="button" onClick={() => fileInputRef.current?.click()} className="flex h-[180px] w-full items-center justify-center overflow-hidden rounded-3xl border border-dashed border-stone-300 bg-stone-50 p-3 text-center transition hover:border-stone-400 hover:bg-stone-100 md:h-[220px]">
            {data.photo ? (
              <img src={data.photo} alt="個案照片" className="h-full w-full rounded-2xl object-contain" />
            ) : (
              <div>
                <div className="text-3xl">📷</div>
                <div className="mt-2 text-sm font-medium text-stone-700">先上傳正面照</div>
                <div className="mt-1 text-xs text-stone-400">照片會固定在上方方便比對</div>
              </div>
            )}
          </button>
        </div>

        <div>
          <div className="mb-3">
            <div className="text-sm font-semibold text-stone-900">目前判斷摘要</div>
            <div className="mt-1 text-xs text-stone-400">操作 FS / RT / AG / LC / VM / ST 時可隨時確認</div>
          </div>

          <div className="grid gap-3 md:grid-cols-2">
            {summaryRows.map(([label, value, targetStep]) => (
              <button key={label} type="button" onClick={() => setCurrent(targetStep)} className="flex items-center justify-between rounded-3xl bg-stone-50 px-5 py-4 text-left text-sm transition hover:bg-stone-100">
                <span className="text-stone-400">{label}</span>
                <span className="font-medium text-stone-800">{value}</span>
              </button>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}

function BasicPanel({ data, setData, aiState, onAIAnalyze, onSyncAll, apiStates, landmarkState, onLandmarkMeasure }) {
  const measurementResult = useMemo(() => calculateFaceMeasurements(data.measurements || {}), [data.measurements]);
  const isSyncing = Object.values(apiStates).some((s) => s.status === "loading");

  const updateMeasurement = (key, value) => setData((prev) => ({ ...prev, measurements: { ...(prev.measurements || {}), [key]: value } }));
  const updateField = (key, value) => setData((prev) => ({ ...prev, [key]: value }));


  return (
    <section className="space-y-6">
      <header className="flex items-start justify-between">
        <div>
          <h2 className="text-2xl font-semibold text-stone-900">AI 判讀與量測</h2>
          <p className="mt-2 text-stone-500">上傳照片後，再進行 AI 初判、點位偵測與人工修正。</p>
        </div>
        <button type="button" onClick={onSyncAll} disabled={isSyncing} className={cx("flex h-10 w-10 items-center justify-center rounded-full border transition", isSyncing ? "border-stone-300 bg-stone-100" : "border-stone-200 bg-white hover:bg-stone-50")}>
          <span className={cx("text-lg", isSyncing && "animate-spin")}>⟳</span>
        </button>
      </header>

      <div className="rounded-3xl border border-stone-200 bg-white p-5 shadow-sm">
        <div className="mb-2 text-sm font-semibold text-stone-900">AI 輔助判讀</div>
        <div className="mb-4 text-xs leading-5 text-stone-500">
          先依照片輔助判斷臉型、比例、直曲、量感、風格與年齡感；三庭比例請使用點位偵測或人工量測確認。
        </div>

        <div className="mb-4 rounded-3xl border border-rose-100 bg-rose-50/40 p-4">
          <SingleChoiceGroup
            label="AI 模型選擇"
            value={data.aiProvider || "gemini"}
            options={["gemini", "openai"]}
            onChange={(next) => updateField("aiProvider", next || "gemini")}
          />
          <div className="mt-2 text-xs leading-5 text-stone-500">
            Gemini 為目前穩定預設；OpenAI 可用來做同張照片比較判讀。
          </div>
        </div>

        <div className="flex flex-wrap gap-3">
          <button type="button" disabled={!data.photo || aiState.status === "loading"} onClick={onAIAnalyze} className="rounded-full bg-stone-900 px-5 py-2 text-sm text-white disabled:cursor-not-allowed disabled:opacity-30">
            {aiState.status === "loading" ? "AI 分析中…" : "AI 初判臉型／比例／直曲／量感／風格／年齡感"}
          </button>

          <button type="button" disabled={!data.photo || landmarkState.status === "loading"} onClick={onLandmarkMeasure} className="rounded-full border border-stone-200 bg-white px-5 py-2 text-sm text-stone-700 hover:bg-stone-100 disabled:cursor-not-allowed disabled:opacity-30">
            {landmarkState.status === "loading" ? "點位偵測中…" : "點位偵測三庭比例"}
          </button>
        </div>

        {aiState.error && <div className="mt-3 rounded-2xl bg-rose-50 px-4 py-3 text-sm leading-6 text-rose-700">{aiState.error}</div>}
        {landmarkState.error && <div className="mt-3 rounded-2xl bg-rose-50 px-4 py-3 text-sm text-rose-700">{landmarkState.error}</div>}
        {landmarkState.status === "success" && <div className="mt-3 rounded-2xl bg-stone-50 px-4 py-3 text-xs leading-5 text-stone-500">已完成點位偵測：系統已依臉部關鍵點估算臉長、臉寬與三庭比例。此數值為輔助參考，請依照片狀態與顧問判斷微調後再儲存。</div>}

        {aiState.result && (
          <div className="mt-4 rounded-2xl bg-stone-50 px-4 py-3 text-sm leading-6 text-stone-700">
            <div>AI 初判：{aiState.result.faceShape || "未判斷"}｜{aiState.result.ratio || "未判斷"}｜{aiState.result.line || "未判斷"}｜{aiState.result.volume || "未判斷"}｜{aiState.result.style || "未判斷"}｜{aiState.result.age || "未判斷"}</div>
          </div>
        )}
      </div>

      <div className="rounded-3xl border border-stone-200 bg-white p-5 shadow-sm">
        <div className="mb-4">
          <div className="text-sm font-semibold text-stone-900">點位估算／人工量測</div>
          <div className="mt-1 text-xs text-stone-500">可先用點位偵測取得比例參考，再由顧問依照片人工修正；若手動輸入 cm 或 mm，同一位個案請統一單位。</div>
        </div>
        <div className="grid gap-4 md:grid-cols-2">
          {["faceLength", "faceWidth", "upperThird", "middleThird", "lowerThird"].map((key) => (
            <Field key={key} label={{ faceLength: "臉長", faceWidth: "臉寬", upperThird: "上庭長度", middleThird: "中庭長度", lowerThird: "下庭長度" }[key]}>
              <input type="number" value={data.measurements?.[key] || ""} onChange={(e) => updateMeasurement(key, e.target.value)} className="w-full rounded-2xl border border-rose-200 bg-rose-50 px-4 py-3 outline-none focus:border-rose-400" />
            </Field>
          ))}
        </div>

        <div className="mt-5 grid gap-3 md:grid-cols-2">
          <div className="rounded-2xl bg-stone-50 px-4 py-3 text-sm">
            <div className="text-stone-500">臉長寬比</div>
            <div className="mt-1 font-semibold text-stone-900">{measurementResult.lengthWidthRatio ? measurementResult.lengthWidthRatio.toFixed(2) : "-"}</div>
            <div className="mt-1 text-xs text-stone-500">{measurementResult.faceRatioLabel}</div>
          </div>
          <div className="rounded-2xl bg-stone-50 px-4 py-3 text-sm">
            <div className="text-stone-500">三庭比例</div>
            <div className="mt-1 font-semibold text-stone-900">{measurementResult.upperPct ? `${measurementResult.upperPct.toFixed(0)} : ${measurementResult.middlePct.toFixed(0)} : ${measurementResult.lowerPct.toFixed(0)}` : "-"}</div>
            <div className="mt-1 text-xs text-stone-500">{measurementResult.thirdsLabel}</div>
          </div>
        </div>

        <div className="mt-5">
          <SingleSelectTagGroup title="色彩整合補充" groups={colorSeasonGroups} value={data.colorSeason || ""} onChange={(next) => updateField("colorSeason", next)} />
        </div>

        <div className="mt-5">
          <Field label="額外補充">
            <textarea value={data.extraObservation || ""} onChange={(e) => updateField("extraObservation", e.target.value)} className="min-h-28 w-full rounded-2xl border border-rose-200 bg-rose-50 px-4 py-3 outline-none focus:border-rose-400" placeholder="例如：可保留 Fresh 感，但妝感不宜過透明。" />
          </Field>
        </div>
      </div>
    </section>
  );
}

function AnalysisPanel({ title, description, label, options, value, onChange, apiState, onReload, children, childrenBefore = null, selectorOverride = null }) {
  return (
    <section className="space-y-6">
      <header>
        <h2 className="text-2xl font-semibold text-stone-900">{title}</h2>
        <p className="mt-2 text-stone-500">{description}</p>
      </header>
      <ApiNotice apiState={apiState} />
      {childrenBefore}
      <Field label={label}>
        {selectorOverride || <OptionGrid options={options} value={value} onChange={onChange} loading={apiState.status === "loading" && options.length === 0} />}
      </Field>
      {children}
      <button type="button" onClick={onReload} className="rounded-full border border-stone-200 px-4 py-2 text-sm text-stone-600 hover:bg-stone-50">重新讀取此分頁</button>
    </section>
  );
}

function buildConsultantReport(data, options, generated) {
  if (generated) {
    return {
      title: `${getName(options.face, data.face)} × ${getName(options.ratio, data.ratio)} × ${getName(options.style, data.style)}`,
      summary: generated.summary || "",
      sections: [
        { title: "造型建議", items: [generated.styling || "尚未生成"] },
        { title: "眼鏡建議", items: [generated.glasses || "尚未生成"] },
        { title: "耳環建議", items: [generated.earrings || "尚未生成"] },
        { title: "妝容建議", items: [generated.makeup || "尚未生成"] },
      ],
    };
  }

  return {
    title: `${getName(options.face, data.face)} × ${getName(options.ratio, data.ratio)} × ${getName(options.style, data.style)}`,
    summary: `這位個案目前判定為 ${getName(options.face, data.face)}，比例狀態為 ${getName(options.ratio, data.ratio)}，直曲線屬於 ${getName(options.line, data.line)}，量感為 ${getName(options.volume, data.volume)}，整體風格定位為 ${getName(options.style, data.style)}。`,
    sections: [
      { title: "目前說明", items: ["尚未生成 S～W，請先儲存個案後再按生成整體建議。"] },
    ],
  };
}

function ConsultantReportCard({ report }) {
  return (
    <div className="rounded-3xl border border-stone-200 bg-stone-50 p-6">
      <div className="mb-5">
        <div className="text-sm font-medium text-stone-500">顧問報告草稿</div>
        <h3 className="mt-1 text-xl font-semibold text-stone-900">{report.title}</h3>
        <p className="mt-3 leading-7 text-stone-700">{report.summary}</p>
      </div>
      <div className="space-y-4">
        {report.sections.map((section) => (
          <div key={section.title} className="rounded-3xl bg-white p-5 shadow-sm">
            <div className="mb-3 text-sm font-semibold text-stone-900">{section.title}</div>
            <div className="space-y-2">
              {section.items.map((item, index) => <div key={`${section.title}-${index}`} className="rounded-2xl bg-stone-50 px-4 py-3 text-sm leading-6 text-stone-700">{item}</div>)}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function buildDirectionNotes(data, options) {
  const records = {
    face: getRecord(options.face, data.face),
    ratio: getRecord(options.ratio, data.ratio),
    line: getRecord(options.line, data.line),
    volume: getRecord(options.volume, data.volume),
    style: getRecord(options.style, data.style),
    age: getRecord(options.age, data.age),
  };

  const notes = [];
  const push = (label, value) => {
    if (value) notes.push(`${label}：${value}`);
  };

  push("臉型修飾", records.face?.direction);
  push("比例修飾", records.ratio?.direction);
  push("線條方向", records.line?.direction);
  push("量感修飾", records.volume?.direction);
  push("風格方向", records.style?.direction);
  push("年齡感方向", records.age?.direction);

  const collect = (key) => [records.face?.[key], records.ratio?.[key], records.line?.[key], records.volume?.[key], records.style?.[key], records.age?.[key]].filter(Boolean).join("｜");

  push("髮型策略", collect("hairDirection"));
  push("瀏海策略", collect("bangDirection"));
  push("分線策略", collect("partDirection"));
  push("眼鏡策略", collect("glassesDirection"));
  push("耳環策略", collect("earringsDirection"));
  push("妝容策略", collect("makeupDirection"));
  push("避免策略", [collect("avoidDirection"), records.line?.avoidHairDirection].filter(Boolean).join("｜"));

  return notes.length ? notes : ["請完成前方分析，系統會在這裡整理修飾方向。"];
}

function RecommendPanel({ data, options, saveState, recommendState, onGenerateRecommendation }) {
  const report = useMemo(() => buildConsultantReport(data, options, recommendState?.result), [data, options, recommendState?.result]);
  const directionNotes = useMemo(() => buildDirectionNotes(data, options), [data, options]);
  const summary = [
    `臉型：${getName(options.face, data.face)}`,
    `比例：${getName(options.ratio, data.ratio)}`,
    `直曲：${getName(options.line, data.line)}`,
    `量感：${getName(options.volume, data.volume)}`,
    `風格：${getName(options.style, data.style)}`,
    `年齡感：${getName(options.age, data.age)}`,
  ];
  return (
    <section className="space-y-6">
      <header className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
        <div>
          <h2 className="text-2xl font-semibold text-stone-900">建議輸出</h2>
          <p className="mt-2 text-stone-500">先儲存個案，再生成整體建議並寫回 01 客戶分析 S～W 欄。</p>
        </div>
        <button onClick={onGenerateRecommendation} disabled={!saveState?.rowNumber || recommendState?.status === "loading"} className="rounded-full bg-stone-900 px-5 py-2 text-sm text-white disabled:cursor-not-allowed disabled:opacity-40">
          {recommendState?.status === "loading" ? "生成中…" : "生成整體建議"}
        </button>
      </header>
      {!saveState?.rowNumber && <div className="rounded-2xl bg-rose-50 px-4 py-3 text-sm text-rose-700">請先在右側「目前判斷摘要」儲存個案到 01。</div>}
      {recommendState?.error && <div className="rounded-2xl bg-rose-50 px-4 py-3 text-sm text-rose-700">{recommendState.error}</div>}
      {recommendState?.status === "success" && <div className="rounded-2xl bg-emerald-50 px-4 py-3 text-sm text-emerald-700">已生成並寫入 01 客戶分析 S～W 欄。</div>}
      <div className="rounded-3xl border border-stone-200 bg-white p-5"><div className="mb-4 text-sm font-semibold text-stone-800">分析摘要</div><div className="grid gap-3 md:grid-cols-2">{summary.map((item) => <div key={item} className="rounded-2xl bg-stone-50 px-4 py-3 text-sm text-stone-700">{item}</div>)}</div></div>

      <div className="rounded-3xl border border-stone-200 bg-white p-5">
        <div className="mb-4 text-sm font-semibold text-stone-800">修飾目標標籤</div>
        <div className="flex flex-wrap gap-2">
          {(data.correctionGoalTags || []).length ? (data.correctionGoalTags || []).map((tag) => (
            <span key={tag} className="rounded-full bg-rose-50 px-4 py-2 text-sm text-rose-700">{tag}</span>
          )) : <div className="text-sm text-stone-400">尚未選擇修飾目標標籤</div>}
        </div>
      </div>

      <div className="rounded-3xl border border-stone-200 bg-white p-5">
        <div className="mb-4 text-sm font-semibold text-stone-800">修飾方向</div>
        <div className="space-y-2">{directionNotes.map((note) => <div key={note} className="rounded-2xl bg-stone-50 px-4 py-3 text-sm leading-6 text-stone-700">{note}</div>)}</div>
      </div>

      <ConsultantReportCard report={report} />
    </section>
  );
}

function InsightPanel({ data, options, apiStates, saveState, onSaveCustomer, setData, setSaveState, setRecommendState, setCurrent }) {
  const selected = useMemo(() => ({
    face: getName(options.face, data.face), ratio: getName(options.ratio, data.ratio), line: getName(options.line, data.line), volume: getName(options.volume, data.volume), style: getName(options.style, data.style), age: getName(options.age, data.age),
  }), [data, options]);
  const records = useMemo(() => ({
    face: getRecord(options.face, data.face), ratio: getRecord(options.ratio, data.ratio), line: getRecord(options.line, data.line), volume: getRecord(options.volume, data.volume), style: getRecord(options.style, data.style), age: getRecord(options.age, data.age),
  }), [data, options]);
  const direction = useMemo(() => {
    const notes = [];
    const push = (label, value) => {
      if (value) notes.push(`${label}：${value}`);
    };

    push("臉型修飾", records.face?.direction);
    push("比例修飾", records.ratio?.direction);
    push("線條方向", records.line?.direction);
    push("量感修飾", records.volume?.direction);
    push("風格方向", records.style?.direction);
    push("年齡感方向", records.age?.direction);

    const hairSet = [records.face?.hairDirection, records.ratio?.hairDirection, records.line?.hairDirection, records.volume?.hairDirection, records.style?.hairDirection, records.age?.hairDirection].filter(Boolean);
    const bangSet = [records.face?.bangDirection, records.ratio?.bangDirection, records.line?.bangDirection, records.volume?.bangDirection, records.style?.bangDirection, records.age?.bangDirection].filter(Boolean);
    const partSet = [records.face?.partDirection, records.ratio?.partDirection, records.line?.partDirection, records.volume?.partDirection, records.style?.partDirection, records.age?.partDirection].filter(Boolean);
    const glassesSet = [records.face?.glassesDirection, records.ratio?.glassesDirection, records.line?.glassesDirection, records.volume?.glassesDirection, records.style?.glassesDirection, records.age?.glassesDirection].filter(Boolean);
    const earringsSet = [records.face?.earringsDirection, records.ratio?.earringsDirection, records.line?.earringsDirection, records.volume?.earringsDirection, records.style?.earringsDirection, records.age?.earringsDirection].filter(Boolean);
    const makeupSet = [records.face?.makeupDirection, records.ratio?.makeupDirection, records.line?.makeupDirection, records.volume?.makeupDirection, records.style?.makeupDirection, records.age?.makeupDirection].filter(Boolean);
    const avoidSet = [records.face?.avoidDirection, records.ratio?.avoidDirection, records.line?.avoidHairDirection, records.line?.avoidDirection, records.volume?.avoidDirection, records.style?.avoidDirection, records.age?.avoidDirection].filter(Boolean);

    push("髮型策略", hairSet.join("｜"));
    push("瀏海策略", bangSet.join("｜"));
    push("分線策略", partSet.join("｜"));
    push("眼鏡策略", glassesSet.join("｜"));
    push("耳環策略", earringsSet.join("｜"));
    push("妝容策略", makeupSet.join("｜"));
    push("避免策略", avoidSet.join("｜"));

    return notes.length ? notes : ["請完成左側分析，系統會整理目前修飾方向。"];
  }, [records]);
  const connectedCount = Object.values(apiStates).filter((state) => state.status === "success").length;

  return (
    <aside className="sticky top-8 h-fit rounded-[2rem] border border-stone-200 bg-white p-5 shadow-sm">
      <div className="mb-4 flex items-center justify-between">
        <div><div className="text-sm text-stone-500">控制中心</div><h3 className="text-lg font-semibold text-stone-900">目前判斷摘要</h3></div>
        <span className="rounded-full bg-stone-900 px-3 py-1 text-xs text-white">{connectedCount}/6 已讀取</span>
      </div>

      <div className="mb-4 rounded-3xl border border-stone-100 bg-stone-50 p-4">
        <div className="mb-2 text-sm font-semibold text-stone-900">儲存個案</div>
        <p className="mb-3 text-xs leading-5 text-stone-500">人工確認所有分析與觀察資料後，先儲存到 01，再生成整體建議。</p>
        <button type="button" disabled={saveState?.status === "loading"} onClick={onSaveCustomer} className="w-full rounded-full bg-stone-900 px-4 py-2 text-sm text-white disabled:cursor-not-allowed disabled:opacity-30">
          {saveState?.status === "loading" ? "儲存中…" : "儲存個案到 01"}
        </button>
        {saveState?.status === "success" && <div className="mt-3 rounded-2xl bg-emerald-50 px-3 py-2 text-xs leading-5 text-emerald-700">已儲存至第 {saveState.rowNumber} 列。</div>}
        {saveState?.error && <div className="mt-3 rounded-2xl bg-rose-50 px-3 py-2 text-xs leading-5 text-rose-700">{saveState.error}</div>}
      </div>

      {data.photo && <div className="mb-5 rounded-3xl bg-stone-50 p-2"><img src={data.photo} alt="個案" className="max-h-96 w-full rounded-2xl object-contain" /></div>}

      <div className="space-y-3">
        {[
          ["臉型", selected.face, 1],
          ["比例", selected.ratio, 2],
          ["年齡感", selected.age, 3],
          ["直曲", selected.line, 4],
          ["量感", selected.volume, 5],
          ["風格", selected.style, 6],
          ["色彩季型補充", data.colorSeason || "未選擇", 0],
        ].map(([label, value, targetStep]) => (
          <button
            key={label}
            type="button"
            onClick={() => setCurrent(targetStep)}
            className="flex w-full items-center justify-between rounded-2xl bg-stone-50 px-4 py-3 text-left text-sm transition hover:bg-stone-100"
          >
            <span className="text-stone-500">{label}</span>
            <span className="font-medium text-stone-900">{value}</span>
          </button>
        ))}
      </div>

      <div className="mt-5 rounded-3xl border border-stone-100 bg-white p-4">
        <div className="mb-3 text-sm font-semibold text-stone-800">修飾目標標籤</div>
        <div className="flex flex-wrap gap-2">
          {observationTagGroups.correctionGoalTags.options.map((tag) => {
            const selectedTags = data.correctionGoalTags || [];
            const active = selectedTags.includes(tag);
            return (
              <button key={tag} type="button" onClick={() => {
                const next = active ? selectedTags.filter((item) => item !== tag) : [...selectedTags, tag];
                setData({ ...data, correctionGoalTags: next });
                setSaveState({ status: "idle", error: "", rowNumber: null, customerId: "" });
                setRecommendState({ status: "idle", error: "", result: null });
              }} className={cx("rounded-full border px-3 py-2 text-xs transition", active ? "border-rose-700 bg-rose-600 text-white shadow-sm" : "border-rose-200 bg-rose-50 text-stone-700 hover:border-rose-400 hover:bg-rose-100")}>
                {active ? "✓ " : ""}{tag}
              </button>
            );
          })}
        </div>
      </div>

      <div className="mt-5 rounded-3xl bg-white p-4">
        <div className="mb-2 text-sm font-semibold text-stone-800">修飾方向</div>
        <div className="space-y-2">{direction.map((note) => <div key={note} className="rounded-2xl bg-stone-50 px-3 py-2 text-sm leading-6 text-stone-700">{note}</div>)}</div>
      </div>
    </aside>
  );
}

export default function App() {
  const [current, setCurrent] = useState(1);
  const contentTopRef = useRef(null);

  const jumpToStep = (step) => {
    setCurrent(step);
    setTimeout(() => {
      contentTopRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    }, 0);
  };
  const [options, setOptions] = useState(fallbackOptions);
  const [apiStates, setApiStates] = useState({ face: { status: "idle", error: "" }, ratio: { status: "idle", error: "" }, line: { status: "idle", error: "" }, volume: { status: "idle", error: "" }, style: { status: "idle", error: "" }, age: { status: "idle", error: "" } });
  const [data, setData] = useState({
    clientName: "", need: "完整形象報告", photo: null, aiProvider: "gemini",
    measurements: { faceLength: "", faceWidth: "", upperThird: "", middleThird: "", lowerThird: "" },
    eyeObservation: "", extraObservation: "", colorSeason: "",
    ageAssessment: {}, lineAssessment: {},
    eyeDistance: "", featureDistribution: "", faceBlank: "", visualFocus: "",
    faceDetailTags: [], ratioFocusTags: [], featureStructureTags: [], correctionGoalTags: [], styleSupplementTags: [],
    face: "", ratio: "", line: "", volume: "", style: "", age: "",
  });
  const [aiState, setAiState] = useState({ status: "idle", error: "", result: null, mapped: null });
  const [landmarkState, setLandmarkState] = useState({ status: "idle", error: "" });
  const [saveState, setSaveState] = useState({ status: "idle", error: "", rowNumber: null, customerId: "" });
  const [recommendState, setRecommendState] = useState({ status: "idle", error: "", result: null });

  const resetSave = () => {
    setSaveState({ status: "idle", error: "", rowNumber: null, customerId: "" });
    setRecommendState({ status: "idle", error: "", result: null });
  };

  const reloadOptionGroup = async (type) => {
    const config = sheetConfig[type];
    if (!config) return false;
    setApiStates((prev) => ({ ...prev, [type]: { status: "loading", error: "" } }));
    try {
      const rawRows = await fetchSheetRows(config.sheet);
      const normalized = normalizeRows(rawRows, type);
      if (normalized.length > 0) {
        setOptions((prev) => ({ ...prev, [type]: normalized }));
        setApiStates((prev) => ({ ...prev, [type]: { status: "success", error: "" } }));
        return true;
      }
      throw new Error("試算表沒有有效資料。");
    } catch (err) {
      setApiStates((prev) => ({ ...prev, [type]: { status: "error", error: String(err) } }));
      return false;
    }
  };

  const handleSyncAll = async () => {
    for (const type of Object.keys(sheetConfig)) await reloadOptionGroup(type);
  };

  useEffect(() => {
    handleSyncAll();
  }, []);

  const handleLandmarkMeasure = async () => {
    if (!data.photo) return;
    setLandmarkState({ status: "loading", error: "" });

    try {
      const img = await loadImageFromBase64(data.photo);
      const landmarker = await getFaceLandmarker();
      const detection = landmarker.detect(img);
      const landmarks = detection?.faceLandmarks?.[0];

      if (!landmarks || !landmarks.length) {
        throw new Error("沒有偵測到清楚臉部點位，請換一張正面、光線充足、臉部完整的照片。 ");
      }

      const measured = estimateMeasurementsFromLandmarks(
        landmarks,
        img.naturalWidth || img.width,
        img.naturalHeight || img.height
      );

      const measurementResult = calculateFaceMeasurements(measured);
      let focusTag = "視覺重心平衡";
      if (measurementResult.upperPct && measurementResult.middlePct && measurementResult.lowerPct) {
        const values = [
          { tag: "視覺重心偏上", value: measurementResult.upperPct },
          { tag: "視覺重心平衡", value: measurementResult.middlePct },
          { tag: "視覺重心偏下", value: measurementResult.lowerPct },
        ];
        const maxItem = values.sort((a, b) => b.value - a.value)[0];
        const minValue = Math.min(measurementResult.upperPct, measurementResult.middlePct, measurementResult.lowerPct);
        focusTag = maxItem.value - minValue <= 6 ? "視覺重心平衡" : maxItem.tag;
      }

      setData((prev) => ({
        ...prev,
        measurements: {
          ...(prev.measurements || {}),
          faceLength: measured.faceLength || prev.measurements?.faceLength || "",
          faceWidth: measured.faceWidth || prev.measurements?.faceWidth || "",
          upperThird: measured.upperThird || prev.measurements?.upperThird || "",
          middleThird: measured.middleThird || prev.measurements?.middleThird || "",
          lowerThird: measured.lowerThird || prev.measurements?.lowerThird || "",
        },
        ratioFocusTags: [focusTag],
      }));

      setLandmarkState({ status: "success", error: "" });
      resetSave();
    } catch (error) {
      setLandmarkState({ status: "error", error: formatClientError(error) });
    }
  };

  const handleAIAnalyze = async () => {
    if (!data.photo) return;
    setAiState({ status: "loading", error: "", result: null, mapped: null });
    try {
      const result = await requestAIInitialAnalysis(data.photo, data.aiProvider || "gemini");
      const mapped = mapAIResultToCodes(result);
      setData((prev) => ({
        ...prev,
        ...(mapped.face ? { face: mapped.face } : {}),
        ...(mapped.ratio ? { ratio: mapped.ratio } : {}),
        ...(mapped.line ? { line: mapped.line } : {}),
        ...(mapped.volume ? { volume: mapped.volume } : {}),
        ...(mapped.style ? { style: mapped.style } : {}),
        ...(mapped.age ? { age: mapped.age } : {}),
      }));
      setAiState({ status: "success", error: "", result, mapped });
      resetSave();
    } catch (error) {
      setAiState({ status: "error", error: formatClientError(error), result: null, mapped: null });
    }
  };

  const handleSaveCustomer = async () => {
    setSaveState({ status: "loading", error: "", rowNumber: null, customerId: "" });
    setRecommendState({ status: "idle", error: "", result: null });
    try {
      const measurementResult = calculateFaceMeasurements(data.measurements || {});
      const thirdsRatio = measurementResult.upperPct ? `${measurementResult.upperPct.toFixed(0)} : ${measurementResult.middlePct.toFixed(0)} : ${measurementResult.lowerPct.toFixed(0)}` : "";
      const eyeObservationForSave = [
        data.eyeDistance ? `眼間距${data.eyeDistance}` : "",
        data.featureDistribution ? `五官${data.featureDistribution}` : "",
        data.faceBlank ? `臉部留白${data.faceBlank}` : "",
        data.visualFocus ? `視覺重心在${data.visualFocus}` : "",
      ].filter(Boolean).join("、") || data.eyeObservation || "";

      const extraObservationForGemini = [data.colorSeason ? `色彩季型補充：${data.colorSeason}` : "", data.styleSupplementTags?.[0] ? `副風格：${data.styleSupplementTags[0]}` : "", data.extraObservation || ""].filter(Boolean).join("\n");
      const payload = {
        action: "saveCustomer",
        data: {
          name: data.clientName || "未命名個案",
          date: new Date().toISOString().slice(0, 10),
          faceName: getName(options.face, data.face), faceCode: data.face,
          ratioName: getName(options.ratio, data.ratio), ratioCode: data.ratio,
          lineName: getName(options.line, data.line), lineCode: data.line,
          volumeName: getName(options.volume, data.volume), volumeCode: data.volume,
          ageName: getName(options.age, data.age), ageCode: data.age,
          styleName: getName(options.style, data.style), styleCode: data.style,
          note: "",
          faceLength: data.measurements?.faceLength || "",
          faceWidth: data.measurements?.faceWidth || "",
          lengthWidthRatio: measurementResult.lengthWidthRatio ? measurementResult.lengthWidthRatio.toFixed(2) : "",
          upperThird: data.measurements?.upperThird || "",
          middleThird: data.measurements?.middleThird || "",
          lowerThird: data.measurements?.lowerThird || "",
          thirdsRatio,
          eyeObservation: eyeObservationForSave,
          extraObservation: extraObservationForGemini,
          faceDetailTags: (data.faceDetailTags || []).join("、"),
          ratioFocusTags: (data.ratioFocusTags || []).join("、"),
          featureStructureTags: (data.featureStructureTags || []).join("、"),
          correctionGoalTags: (data.correctionGoalTags || []).join("、"),
          ageAssessmentText: buildAgeAssessmentText(data.ageAssessment || {}),
          ageAssessmentResult: getAgeAssessmentResult(data.ageAssessment || {}),
          lineAssessmentText: buildLineAssessmentText(data.lineAssessment || {}),
          lineAssessmentResult: getLineAssessmentResult(data.lineAssessment || {}),
        },
      };
      const response = await fetch(API_BASE_URL, { method: "POST", body: JSON.stringify(payload) });
      const result = await response.json();
      if (!result.ok) throw new Error(result.error || "儲存失敗");
      setSaveState({ status: "success", error: "", rowNumber: result.result?.rowNumber, customerId: result.result?.customerId || "" });
    } catch (error) {
      setSaveState({ status: "error", error: formatClientError(error), rowNumber: null, customerId: "" });
    }
  };

  const handleGenerateRecommendation = async () => {
    if (!saveState.rowNumber) return;
    setRecommendState({ status: "loading", error: "", result: null });
    try {
      const response = await fetch(API_BASE_URL, { method: "POST", body: JSON.stringify({ action: "generateRecommendation", row: saveState.rowNumber, provider: data.aiProvider || "gemini" }) });
      const result = await response.json();
      if (!result.ok) throw new Error(result.error || "生成失敗");
      setRecommendState({ status: "success", error: "", result: result.result });
    } catch (error) {
      setRecommendState({ status: "error", error: formatClientError(error), result: null });
    }
  };

  const handleMainChange = (key, val) => {
    setData({ ...data, [key]: val });
    resetSave();
  };

  return (
    <div className="min-h-screen bg-stone-50 p-4 font-sans text-stone-900 md:p-8 selection:bg-rose-500 selection:text-white">
      <div className="mx-auto max-w-7xl">
        <ProgressRail current={current} setCurrent={jumpToStep} data={data} setData={setData} />

        <div className="grid gap-8 lg:grid-cols-[380px_1fr]">
          <aside className="space-y-6 lg:sticky lg:top-6 lg:h-[calc(100vh-3rem)] lg:overflow-y-auto">
            <StickyPhotoSummaryPanel data={data} setData={setData} options={options} apiStates={apiStates} saveState={saveState} onSaveCustomer={handleSaveCustomer} setCurrent={jumpToStep} />
            <BasicPanel data={data} setData={setData} aiState={aiState} onAIAnalyze={handleAIAnalyze} onSyncAll={handleSyncAll} apiStates={apiStates} landmarkState={landmarkState} onLandmarkMeasure={handleLandmarkMeasure} />
          </aside>

          <main className="min-h-[650px] rounded-[2.5rem] border border-stone-100 bg-white p-6 shadow-2xl shadow-stone-200/60 md:p-10">
          <div ref={contentTopRef} className="scroll-mt-32" />
          <div className="mb-8 flex items-center justify-between">
            <button type="button" onClick={() => jumpToStep(Math.max(1, current - 1))} disabled={current <= 1} className="rounded-full border border-stone-200 bg-white px-4 py-2 text-sm text-stone-600 hover:bg-stone-50 disabled:cursor-not-allowed disabled:opacity-30">← 上一步</button>
            <div className="rounded-full bg-stone-100 px-4 py-2 text-xs font-semibold text-stone-600">{current} / {steps.length - 1}</div>
            <button type="button" onClick={() => jumpToStep(Math.min(steps.length - 1, current + 1))} disabled={current === steps.length - 1} className="rounded-full border border-stone-200 bg-white px-4 py-2 text-sm text-stone-600 hover:bg-stone-50 disabled:cursor-not-allowed disabled:opacity-30">下一步 →</button>
          </div>

          {current === 0 && <div className="rounded-3xl bg-stone-50 p-6 text-sm leading-6 text-stone-600">基本資料已固定在左側，請從上方選擇臉型 FS 開始分析。</div>}
          {current === 1 && <AnalysisPanel title="臉型分析 FS" description="判斷臉部輪廓與骨架感。" label="選擇臉型" options={options.face} value={data.face} onChange={(val) => handleMainChange("face", val)} apiState={apiStates.face} onReload={() => reloadOptionGroup("face")} selectorOverride={<FaceShapeGrid options={options.face} value={data.face} onChange={(val) => handleMainChange("face", val)} loading={apiStates.face.status === "loading" && options.face.length === 0} />}><GroupedCheckboxTagGroup title={observationTagGroups.faceDetailTags.title} hint={observationTagGroups.faceDetailTags.hint} sections={faceDetailTagSections} value={data.faceDetailTags || []} onChange={(next) => handleMainChange("faceDetailTags", next)} /></AnalysisPanel>}
          {current === 2 && <AnalysisPanel title="比例分析 RT" description="判斷三庭比例、五眼眼距與視覺重心。" label="選擇比例特徵" options={options.ratio} value={data.ratio} onChange={(val) => handleMainChange("ratio", val)} apiState={apiStates.ratio} onReload={() => reloadOptionGroup("ratio")}>
            <div className="space-y-4">
              <div className="rounded-[2rem] border border-stone-200 bg-white p-6 shadow-sm">
                <div className="mb-5 flex items-start gap-4">
                  <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl bg-rose-50 text-xl text-rose-700">◉</div>
                  <div>
                    <div className="text-lg font-semibold text-stone-900">五眼／眼距觀察</div>
                    <div className="mt-1 text-sm leading-6 text-stone-500">此區屬於比例判斷，用來補充眼距、五官分布、臉部留白與視覺重心。</div>
                  </div>
                </div>

                <div className="rounded-[1.75rem] border border-stone-100 bg-stone-50/60 p-5">
                  <div className="grid gap-3">
                    {[
                      ["eyeDistance", "眼間距", "◐", ["較開", "正常", "較近"]],
                      ["featureDistribution", "五官分布", "✦", ["集中", "均勻", "分散"]],
                      ["faceBlank", "臉部留白", "○", ["多", "適中"]],
                      ["visualFocus", "視覺重心", "⌖", ["上", "中", "下"]],
                    ].map(([key, label, icon, items]) => (
                      <div key={key} className="rounded-3xl border border-stone-100 bg-white p-4 shadow-sm md:flex md:items-center md:justify-between md:gap-5">
                        <div className="mb-3 flex items-center gap-3 md:mb-0 md:w-32 md:shrink-0">
                          <div className="flex h-9 w-9 items-center justify-center rounded-full bg-rose-50 text-sm text-rose-700">{icon}</div>
                          <div className="whitespace-nowrap text-base font-semibold text-stone-900">{label}</div>
                        </div>
                        <div className="flex flex-wrap gap-2 md:justify-end">
                          {items.map((item) => {
                            const active = data[key] === item;
                            return (
                              <button
                                key={item}
                                type="button"
                                onClick={() => handleMainChange(key, active ? "" : item)}
                                className={cx(
                                  "min-w-20 rounded-full border px-4 py-2.5 text-center text-sm font-medium transition",
                                  active
                                    ? "border-stone-900 bg-stone-900 text-white shadow-sm"
                                    : "border-rose-200 bg-rose-50/40 text-stone-700 hover:border-rose-400 hover:bg-rose-50"
                                )}
                              >
                                {active ? "✓ " : ""}{item}
                              </button>
                            );
                          })}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="mt-5 rounded-[1.75rem] bg-rose-50/40 p-4">
                  <div className="mb-3 flex items-center gap-2 text-sm font-semibold text-stone-800">
                    <span className="flex h-8 w-8 items-center justify-center rounded-full bg-white text-rose-700">▤</span>
                    觀察文字預覽
                  </div>
                  <div className="rounded-2xl border border-dashed border-rose-200 bg-white px-4 py-3 text-sm leading-6 text-stone-600">
                    {[
                      data.eyeDistance ? `眼間距${data.eyeDistance}` : "",
                      data.featureDistribution ? `五官${data.featureDistribution}` : "",
                      data.faceBlank ? `臉部留白${data.faceBlank}` : "",
                      data.visualFocus ? `視覺重心在${data.visualFocus}` : "",
                    ].filter(Boolean).join("、") || "尚未選擇"}
                  </div>
                </div>
              </div>

              <CheckboxTagGroup title={observationTagGroups.ratioFocusTags.title} hint={observationTagGroups.ratioFocusTags.hint} options={observationTagGroups.ratioFocusTags.options} value={data.ratioFocusTags || []} onChange={(next) => handleMainChange("ratioFocusTags", next)} />
              <CheckboxTagGroup title={observationTagGroups.featureStructureTags.title} hint={observationTagGroups.featureStructureTags.hint} options={observationTagGroups.featureStructureTags.options} value={data.featureStructureTags || []} onChange={(next) => handleMainChange("featureStructureTags", next)} />
            </div>
          </AnalysisPanel>}
          {current === 4 && (
            <AnalysisPanel
              title="直曲分析 LC"
              description="判斷五官與輪廓的線條感。"
              label="選擇直曲類型"
              options={options.line}
              value={data.line}
              onChange={(val) => handleMainChange("line", val)}
              apiState={apiStates.line}
              onReload={() => reloadOptionGroup("line")}
              childrenBefore={
                <BinaryAssessmentTable
                  title="直曲線定位評估表"
                  hint="先依照五個觀察項目勾選直線或曲線，再選擇最終直曲類型。"
                  items={lineAssessmentItems}
                  leftTitle="直線"
                  rightTitle="曲線"
                  leftValue="straight"
                  rightValue="curve"
                  value={data.lineAssessment || {}}
                  onChange={(next) => handleMainChange("lineAssessment", next)}
                  result={getLineAssessmentResult(data.lineAssessment || {})}
                />
              }
            >
              <div className="rounded-3xl border border-stone-200 bg-white p-5 shadow-sm">
                <div className="mb-4 text-sm font-semibold text-stone-900">LC 直曲視覺感說明與造型策略</div>
                <div className="grid gap-4 md:grid-cols-2">
                  <div className="rounded-2xl bg-stone-50 px-4 py-3">
                    <div className="mb-1 text-xs font-medium text-stone-400">視覺特徵／線條感</div>
                    <div className="text-sm leading-6 text-stone-700">
                      {getRecord(options.line, data.line)?.feature || lineVisualDefinitions[data.line] || "請先選擇直曲類型，系統會帶出視覺特徵。"}
                    </div>
                  </div>
                  <div className="rounded-2xl bg-stone-50 px-4 py-3">
                    <div className="mb-1 text-xs font-medium text-stone-400">修飾方向／策略</div>
                    <div className="space-y-2 text-sm leading-6 text-stone-700">
                      {[
                        getRecord(options.line, data.line)?.direction ? `線條方向：${getRecord(options.line, data.line)?.direction}` : "",
                        getRecord(options.line, data.line)?.hairDirection ? `適合髮型：${getRecord(options.line, data.line)?.hairDirection}` : "",
                        getRecord(options.line, data.line)?.avoidHairDirection ? `避免髮型：${getRecord(options.line, data.line)?.avoidHairDirection}` : "",
                        getRecord(options.line, data.line)?.makeupDirection ? `妝容建議：${getRecord(options.line, data.line)?.makeupDirection}` : "",
                      ].filter(Boolean).map((item) => (
                        <div key={item}>{item}</div>
                      ))}
                      {![
                        getRecord(options.line, data.line)?.direction,
                        getRecord(options.line, data.line)?.hairDirection,
                        getRecord(options.line, data.line)?.avoidHairDirection,
                        getRecord(options.line, data.line)?.makeupDirection,
                      ].some(Boolean) && "請先選擇直曲類型，系統會帶出造型策略。"}
                    </div>
                  </div>
                </div>
              </div>
            </AnalysisPanel>
          )}
          {current === 5 && <AnalysisPanel title="量感分析 VM" description="判斷五官的大小與存在感。" label="選擇量感類型" options={options.volume} value={data.volume} onChange={(val) => handleMainChange("volume", val)} apiState={apiStates.volume} onReload={() => reloadOptionGroup("volume")} />}
          {current === 6 && (
            <AnalysisPanel
              title="風格分析 ST"
              description="定位最終的風格坐標。"
              label="選擇主風格"
              options={options.style}
              value={data.style}
              onChange={(val) => handleMainChange("style", val)}
              apiState={apiStates.style}
              onReload={() => reloadOptionGroup("style")}
            >
              <div className="rounded-3xl border border-stone-200 bg-white p-5 shadow-sm">
                <div className="mb-4 flex items-center justify-between gap-4">
                  <div>
                    <div className="text-sm font-semibold text-stone-900">風格座標圖</div>
                    <div className="mt-1 text-xs leading-5 text-stone-500">輔助判斷幼態／成熟、直線／曲線與八大風格位置。</div>
                  </div>
                </div>
                {STYLE_MAP_IMAGE_URL ? (
                  <div className="overflow-hidden rounded-3xl bg-stone-50">
                    <img src={STYLE_MAP_IMAGE_URL} alt="風格座標圖" className="w-full object-contain" />
                  </div>
                ) : (
                  <div className="rounded-3xl border border-dashed border-stone-200 bg-stone-50 px-5 py-8 text-center text-sm leading-6 text-stone-500">
                    請先把風格座標圖上傳成公開圖片網址，<br />再填入 STYLE_MAP_IMAGE_URL。
                  </div>
                )}
              </div>

              <CheckboxTagGroup
                title={observationTagGroups.styleSupplementTags.title}
                hint={observationTagGroups.styleSupplementTags.hint}
                options={observationTagGroups.styleSupplementTags.options}
                value={data.styleSupplementTags || []}
                onChange={(next) => handleMainChange("styleSupplementTags", next.slice(-1))}
              />
            </AnalysisPanel>
          )}
          {current === 3 && (
            <AnalysisPanel
              title="年齡感分析 AG"
              description="判斷視覺年齡的成熟度與整體視覺感。"
              label="選擇年齡感"
              options={options.age}
              value={data.age}
              onChange={(val) => handleMainChange("age", val)}
              apiState={apiStates.age}
              onReload={() => reloadOptionGroup("age")}
              childrenBefore={
                <BinaryAssessmentTable
                  title="年齡感定位評估表"
                  hint="先依照五個觀察項目勾選幼態或成熟，再選擇最終年齡感類型。"
                  items={ageAssessmentItems}
                  leftTitle="幼態"
                  rightTitle="成熟"
                  leftValue="juvenile"
                  rightValue="mature"
                  value={data.ageAssessment || {}}
                  onChange={(next) => handleMainChange("ageAssessment", next)}
                  result={getAgeAssessmentResult(data.ageAssessment || {})}
                />
              }
            >
              <div className="rounded-3xl border border-stone-200 bg-white p-5 shadow-sm">
                <div className="mb-4 text-sm font-semibold text-stone-900">AG 視覺感說明與修飾策略</div>
                <div className="grid gap-4 md:grid-cols-2">
                  <div className="rounded-2xl bg-stone-50 px-4 py-3">
                    <div className="mb-1 text-xs font-medium text-stone-400">視覺特徵／視覺感</div>
                    <div className="text-sm leading-6 text-stone-700">
                      {getRecord(options.age, data.age)?.feature || ageVisualDefinitions[data.age] || "請先選擇年齡感，系統會帶出視覺特徵。"}
                    </div>
                  </div>
                  <div className="rounded-2xl bg-stone-50 px-4 py-3">
                    <div className="mb-1 text-xs font-medium text-stone-400">修飾方向／策略</div>
                    <div className="space-y-2 text-sm leading-6 text-stone-700">
                      {[
                        getRecord(options.age, data.age)?.direction ? `方向：${getRecord(options.age, data.age)?.direction}` : "",
                        getRecord(options.age, data.age)?.hairDirection ? `髮型策略：${getRecord(options.age, data.age)?.hairDirection}` : "",
                        getRecord(options.age, data.age)?.glassesDirection ? `眼鏡策略：${getRecord(options.age, data.age)?.glassesDirection}` : "",
                        getRecord(options.age, data.age)?.earringsDirection ? `耳環策略：${getRecord(options.age, data.age)?.earringsDirection}` : "",
                        getRecord(options.age, data.age)?.makeupDirection ? `妝容策略：${getRecord(options.age, data.age)?.makeupDirection}` : "",
                        getRecord(options.age, data.age)?.avoidDirection ? `避免策略：${getRecord(options.age, data.age)?.avoidDirection}` : "",
                      ].filter(Boolean).map((item) => (
                        <div key={item}>{item}</div>
                      ))}
                      {![
                        getRecord(options.age, data.age)?.direction,
                        getRecord(options.age, data.age)?.hairDirection,
                        getRecord(options.age, data.age)?.glassesDirection,
                        getRecord(options.age, data.age)?.earringsDirection,
                        getRecord(options.age, data.age)?.makeupDirection,
                        getRecord(options.age, data.age)?.avoidDirection,
                      ].some(Boolean) && "請先選擇年齡感，系統會帶出修飾策略。"}
                    </div>
                  </div>
                </div>
              </div>
            </AnalysisPanel>
          )}
          {current === 7 && <RecommendPanel data={data} options={options} saveState={saveState} recommendState={recommendState} onGenerateRecommendation={handleGenerateRecommendation} />}
        </main>
        </div>
      </div>
    </div>
  );
}
