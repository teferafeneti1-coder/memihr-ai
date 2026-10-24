// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const $ = (id) => document.getElementById(id);

const CAT_META = {
  numbers: { label: "Numbers",  emoji: "🔢" },
  body:    { label: "Body",     emoji: "🧠" },
  family:  { label: "Family",   emoji: "👨‍👩‍👧‍👦" },
  animals: { label: "Animals",  emoji: "🐘" },
  nature:  { label: "Nature",   emoji: "🌳" },
  food:    { label: "Food",     emoji: "🍽️" },
  colors:  { label: "Colors",   emoji: "🎨" },
  school:  { label: "School",   emoji: "📚" },
  aksum:   { label: "Aksum",    emoji: "🏛️" },
};

/**
 * Call the model to translate a single word or sentence.
 * Returns the translated string, or null on error.
 */
async function modelTranslate(text, direction) {
  try {
    const res  = await fetch("/api/translate/word", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ text, direction }),
    });
    const data = await res.json();
    return res.ok ? data.translation : null;
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------
// Mode  (adult | kids)
// ---------------------------------------------------------------------------

let currentMode = "adult";

const modeAdultBtn = $("mode-adult");
const modeKidsBtn  = $("mode-kids");

function applyMode(mode) {
  currentMode = mode;
  modeAdultBtn.classList.toggle("is-active", mode === "adult");
  modeKidsBtn.classList.toggle("is-active",  mode === "kids");
  document.body.classList.toggle("kids-mode", mode === "kids");

  $("wordmark-sub").textContent =
    mode === "kids" ? "🌍 learn two languages!" : "a port between two languages";

  $("lessons-adult").hidden = (mode === "kids");
  $("lessons-kids").hidden  = (mode === "adult");
  $("quiz-adult").hidden    = (mode === "kids");
  $("quiz-kids").hidden     = (mode === "adult");

  if (mode === "kids" && !kidsDataLoaded) loadKidsData();
}

modeAdultBtn.addEventListener("click", () => applyMode("adult"));
modeKidsBtn.addEventListener("click",  () => applyMode("kids"));

// ---------------------------------------------------------------------------
// Tabs
// ---------------------------------------------------------------------------

const tabButtons = document.querySelectorAll(".tab-btn");
const panels = {
  translate: $("panel-translate"),
  lessons:   $("panel-lessons"),
  quiz:      $("panel-quiz"),
};

tabButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    tabButtons.forEach((b) => b.classList.remove("is-active"));
    btn.classList.add("is-active");
    Object.values(panels).forEach((p) => p.classList.remove("is-active"));
    panels[btn.dataset.tab].classList.add("is-active");

    if (btn.dataset.tab === "lessons" && currentMode === "adult" && !lessonsLoaded)
      loadLessons();
    if (btn.dataset.tab === "quiz" && currentMode === "adult" && !quizLessonOptionsLoaded)
      loadQuizLessonOptions();
    if ((btn.dataset.tab === "lessons" || btn.dataset.tab === "quiz")
        && currentMode === "kids" && !kidsDataLoaded)
      loadKidsData();
  });
});

// ---------------------------------------------------------------------------
// Translate tab  (shared, unchanged)
// ---------------------------------------------------------------------------

let direction = "am2om";
const srcLabel        = $("src-label");
const tgtLabel        = $("tgt-label");
const srcText         = $("src-text");
const swapBtn         = $("swap-btn");
const translateBtn    = $("translate-btn");
const translateStatus = $("translate-status");
const outputBox       = $("output-box");

const LABELS = { am: "Amharic", om: "Afan Oromo" };
const PLACEHOLDERS = { am2om: "የጥንቷ አክሱም...", om2am: "Aksuumin durii..." };

function applyDirectionUI() {
  const [src, tgt] = direction.split("2");
  srcLabel.textContent = LABELS[src];
  tgtLabel.textContent = LABELS[tgt];
  srcText.setAttribute("lang", src);
  srcText.setAttribute("placeholder", PLACEHOLDERS[direction]);
}

swapBtn.addEventListener("click", () => {
  direction = direction === "am2om" ? "om2am" : "am2om";
  applyDirectionUI();
  srcText.value = "";
  outputBox.innerHTML = '<span class="output-placeholder">Your translation will appear here.</span>';
});

translateBtn.addEventListener("click", async () => {
  const text = srcText.value.trim();
  translateStatus.textContent = "";
  translateStatus.classList.remove("is-error");
  if (!text) {
    translateStatus.textContent = "Type something first.";
    translateStatus.classList.add("is-error");
    return;
  }
  translateBtn.disabled = true;
  translateStatus.textContent = "Translating…";
  try {
    const res  = await fetch("/api/translate", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, direction }),
    });
    const data = await res.json();
    if (!res.ok) {
      translateStatus.textContent = data.error || "Something went wrong.";
      translateStatus.classList.add("is-error");
    } else {
      translateStatus.textContent = "";
      outputBox.innerHTML = "";
      const span = document.createElement("span");
      span.setAttribute("lang", direction === "am2om" ? "om" : "am");
      span.textContent = data.translation;
      outputBox.appendChild(span);
    }
  } catch {
    translateStatus.textContent = "Couldn't reach the server.";
    translateStatus.classList.add("is-error");
  } finally {
    translateBtn.disabled = false;
  }
});

applyDirectionUI();

// ---------------------------------------------------------------------------
// Adult — Lessons  (now with per-card model reveal)
// ---------------------------------------------------------------------------

let lessonsLoaded = false;
const lessonListEl   = $("lesson-list");
const lessonDetailEl = $("lesson-detail");

async function loadLessons() {
  lessonsLoaded = true;
  const res     = await fetch("/api/lessons");
  const lessons = await res.json();
  lessonListEl.innerHTML = "";
  lessons.forEach((lesson) => {
    const li = document.createElement("li");
    li.className   = "lesson-item";
    li.textContent = lesson.title;
    li.dataset.id  = lesson.id;
    li.addEventListener("click", () => openLesson(lesson.id, li));
    lessonListEl.appendChild(li);
  });
}

async function openLesson(id, liEl) {
  document.querySelectorAll(".lesson-item").forEach((el) => el.classList.remove("is-active"));
  liEl.classList.add("is-active");
  lessonDetailEl.innerHTML = "<p class='empty-hint'>Loading…</p>";

  const res    = await fetch(`/api/lessons/${id}`);
  const lesson = await res.json();

  lessonDetailEl.innerHTML = "";

  // Direction selector row
  const dirRow = document.createElement("div");
  dirRow.className = "lesson-dir-row";
  dirRow.innerHTML = `
    <span class="lesson-dir-label">Direction:</span>
    <label class="lesson-dir-opt">
      <input type="radio" name="lesson-dir-${id}" value="am2om" checked> Amharic → Afan Oromo
    </label>
    <label class="lesson-dir-opt">
      <input type="radio" name="lesson-dir-${id}" value="om2am"> Afan Oromo → Amharic
    </label>`;
  lessonDetailEl.appendChild(dirRow);

  const title = document.createElement("h3");
  title.className   = "lesson-title";
  title.textContent = lesson.title;
  lessonDetailEl.appendChild(title);

  lesson.pairs.forEach((pair) => {
    const card = document.createElement("div");
    card.className = `flashcard type-${pair.type}`;

    // Source side (shown by default)
    const srcDiv = document.createElement("div");
    srcDiv.className = "am fc-source";
    srcDiv.setAttribute("lang", "am");
    srcDiv.textContent = pair.am;

    // Model output area (hidden until button clicked)
    const modelDiv = document.createElement("div");
    modelDiv.className = "fc-model-output";
    modelDiv.hidden = true;

    // Reference area (hidden until model runs)
    const refDiv = document.createElement("div");
    refDiv.className = "om fc-reference";
    refDiv.setAttribute("lang", "om");
    refDiv.hidden = true;

    // Reveal button
    const revealBtn = document.createElement("button");
    revealBtn.className   = "reveal-btn";
    revealBtn.textContent = "Translate with model →";

    revealBtn.addEventListener("click", async () => {
      // Read current direction from radio
      const dirInput = dirRow.querySelector(`input[name="lesson-dir-${id}"]:checked`);
      const d = dirInput ? dirInput.value : "am2om";
      const srcKey = d === "am2om" ? "am" : "om";
      const tgtKey = d === "am2om" ? "om" : "am";
      const tgtLang = tgtKey;

      srcDiv.setAttribute("lang", srcKey);
      srcDiv.textContent = pair[srcKey];
      refDiv.setAttribute("lang", tgtLang);
      refDiv.textContent = `Reference: ${pair[tgtKey]}`;

      revealBtn.disabled    = true;
      revealBtn.textContent = "Translating…";
      modelDiv.hidden = true;

      const result = await modelTranslate(pair[srcKey], d);

      modelDiv.hidden = false;
      if (result) {
        modelDiv.innerHTML = `<span class="fc-model-label">Model says:</span>
          <span lang="${tgtLang}">${result}</span>`;
      } else {
        modelDiv.innerHTML = `<span class="fc-model-label is-error">Translation failed.</span>`;
      }

      refDiv.hidden = false;
      revealBtn.textContent = "Re-translate →";
      revealBtn.disabled    = false;
    });

    card.appendChild(srcDiv);
    card.appendChild(revealBtn);
    card.appendChild(modelDiv);
    card.appendChild(refDiv);
    lessonDetailEl.appendChild(card);
  });
}

// ---------------------------------------------------------------------------
// Adult — Quiz  (+ model confirmation after answering)
// ---------------------------------------------------------------------------

let quizLessonOptionsLoaded = false;
const quizLessonSelect = $("quiz-lesson-select");
const quizCountSelect  = $("quiz-count");
const quizStartBtn     = $("quiz-start-btn");
const quizSetup        = $("quiz-setup");
const quizRunner       = $("quiz-runner");
const quizResult       = $("quiz-result");
const quizProgress     = $("quiz-progress");
const quizPromptLabel  = $("quiz-prompt-label");
const quizPrompt       = $("quiz-prompt");
const quizOptions      = $("quiz-options");
const quizFeedback     = $("quiz-feedback");
const quizNextBtn      = $("quiz-next-btn");
const quizScore        = $("quiz-score");
const quizRestartBtn   = $("quiz-restart-btn");

// Model confirmation box — lives below quiz-feedback in the DOM
const quizModelBox = $("quiz-model-box");

let quizQuestions = [], quizIndex = 0, quizCorrectCount = 0;

async function loadQuizLessonOptions() {
  quizLessonOptionsLoaded = true;
  const res     = await fetch("/api/lessons");
  const lessons = await res.json();
  lessons.forEach((lesson) => {
    const opt = document.createElement("option");
    opt.value = lesson.id; opt.textContent = lesson.title;
    quizLessonSelect.appendChild(opt);
  });
}

quizStartBtn.addEventListener("click", async () => {
  quizStartBtn.disabled = true; quizStartBtn.textContent = "Loading…";
  try {
    const res  = await fetch(`/api/quiz?lesson=${quizLessonSelect.value}&count=${quizCountSelect.value}`);
    const data = await res.json();
    if (!res.ok) { alert(data.error || "Could not build a quiz."); return; }
    quizQuestions = data.questions; quizIndex = 0; quizCorrectCount = 0;
    quizSetup.hidden = true; quizResult.hidden = true; quizRunner.hidden = false;
    renderAdultQuizQ();
  } finally { quizStartBtn.disabled = false; quizStartBtn.textContent = "Start quiz"; }
});

async function renderAdultQuizQ() {
  const q       = quizQuestions[quizIndex];
  const tgtLang = q.direction === "am2om" ? "om" : "am";
  const srcLang = q.direction === "am2om" ? "am" : "om";

  quizProgress.textContent = `Question ${quizIndex + 1} of ${quizQuestions.length}`;
  quizPromptLabel.textContent = q.direction === "am2om"
    ? "Translate this Amharic sentence to Afan Oromo:"
    : "Translate this Afan Oromo sentence to Amharic:";
  quizPrompt.setAttribute("lang", srcLang);
  quizPrompt.textContent = q.prompt;

  // Clear everything, hide model box until after user answers
  quizOptions.innerHTML = "";
  quizFeedback.textContent = "";
  quizFeedback.className = "quiz-feedback";
  quizNextBtn.hidden = true;
  if (quizModelBox) { quizModelBox.hidden = true; quizModelBox.innerHTML = ""; }

  // Show loading text in options area while model runs
  const loadingLi = document.createElement("li");
  loadingLi.id = "adult-options-loading";
  loadingLi.innerHTML = `<span class="options-loading-text">⏳ Loading options…</span>`;
  quizOptions.appendChild(loadingLi);

  // Run model first — its output is the correct answer
  const modelOutput = await modelTranslate(q.prompt, q.direction);

  // Remove loading indicator
  const loadingEl = document.getElementById("adult-options-loading");
  if (loadingEl) loadingEl.remove();

  if (!modelOutput) {
    // Model failed — fall back to stored correct option
    if (quizModelBox) quizModelBox.innerHTML =
      `<span class="quiz-model-label is-error">Model unavailable — showing reference options.</span>`;
    q._modelCorrect = q.options[q.correct_index];
  } else {
    q._modelCorrect = modelOutput;
    if (quizModelBox) {
      quizModelBox.innerHTML =
        `<span class="quiz-model-label">Model answer:</span>
         <span lang="${tgtLang}">${modelOutput}</span>`;
    }
  }

  // Build options: model's answer + 3 distractors from the corpus options
  // Use the stored distractors (all options except stored correct), then add model answer
  const distractors = q.options
    .filter((_, i) => i !== q.correct_index)   // the 3 wrong ones from corpus
    .slice(0, 3);

  const opts = [...distractors, q._modelCorrect];
  // Shuffle
  for (let i = opts.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [opts[i], opts[j]] = [opts[j], opts[i]];
  }
  q._opts        = opts;
  q._correctIdx  = opts.indexOf(q._modelCorrect);

  // Render options
  opts.forEach((optText, i) => {
    const li  = document.createElement("li");
    const btn = document.createElement("button");
    btn.className = "quiz-option-btn";
    btn.setAttribute("lang", tgtLang);
    btn.textContent = optText;
    btn.addEventListener("click", () => answerAdultQuiz(i, btn));
    li.appendChild(btn);
    quizOptions.appendChild(li);
  });
}

function answerAdultQuiz(chosenIndex, chosenBtn) {
  const q       = quizQuestions[quizIndex];
  const tgtLang = q.direction === "am2om" ? "om" : "am";
  const buttons = quizOptions.querySelectorAll(".quiz-option-btn");
  buttons.forEach((b) => (b.disabled = true));

  const isCorrect = chosenIndex === q._correctIdx;
  if (isCorrect) quizCorrectCount++;

  chosenBtn.classList.add(isCorrect ? "is-correct" : "is-wrong");
  if (!isCorrect) buttons[q._correctIdx].classList.add("is-correct");

  quizFeedback.textContent = isCorrect
    ? "Correct! That matches the model's translation."
    : "Not quite — the model's translation is highlighted in green.";
  quizFeedback.className = `quiz-feedback ${isCorrect ? "is-correct" : "is-wrong"}`;
  quizNextBtn.hidden = false;

  // Reveal model answer now that user has picked
  if (quizModelBox && q._modelCorrect) {
    quizModelBox.hidden = false;
    quizModelBox.innerHTML =
      `<span class="quiz-model-label">Model says:</span>
       <span lang="${tgtLang}">${q._modelCorrect}</span>`;
  }
}

quizNextBtn.addEventListener("click", () => {
  quizIndex++;
  if (quizIndex >= quizQuestions.length) {
    quizRunner.hidden = true; quizResult.hidden = false;
    quizScore.textContent = `${quizCorrectCount} / ${quizQuestions.length} correct`;
  } else renderAdultQuizQ();
});

quizRestartBtn.addEventListener("click", () => { quizResult.hidden = true; quizSetup.hidden = false; });

// ---------------------------------------------------------------------------
// Kids — shared data
// ---------------------------------------------------------------------------

let kidsDataLoaded = false;
let kidsWordBank   = [];

async function loadKidsData() {
  kidsDataLoaded = true;
  const res  = await fetch("/api/words");
  const data = await res.json();
  kidsWordBank = data.words;
  buildCatGrids(data.categories);
}

function buildCatGrids(categories) {
  [$("kids-cat-grid"), $("kids-quiz-cat-grid")].forEach((grid, gridIdx) => {
    if (!grid) return;
    grid.innerHTML = "";
    grid.appendChild(makeCatTile("mix", "All", "🌍", gridIdx === 0));
    categories.forEach((cat) => {
      const meta = CAT_META[cat] || { label: cat, emoji: "📦" };
      grid.appendChild(makeCatTile(cat, meta.label, meta.emoji, false));
    });
  });

  $("kids-quiz-cat-grid").querySelectorAll(".kids-cat-tile").forEach((tile) => {
    tile.addEventListener("click", () => {
      $("kids-quiz-cat-grid").querySelectorAll(".kids-cat-tile").forEach((t) => t.classList.remove("is-selected"));
      tile.classList.add("is-selected");
    });
  });

  $("kids-cat-grid").querySelectorAll(".kids-cat-tile").forEach((tile) => {
    tile.addEventListener("click", () => openKidsWordCategory(tile.dataset.cat));
  });
}

function makeCatTile(cat, label, emoji, isSelected) {
  const btn = document.createElement("button");
  btn.className = "kids-cat-tile" + (isSelected ? " is-selected" : "");
  btn.dataset.cat = cat;
  btn.innerHTML = `<span class="cat-emoji">${emoji}</span>${label}`;
  return btn;
}

// ---------------------------------------------------------------------------
// Kids — Word Flashcards  (flip calls model live)
// ---------------------------------------------------------------------------

let kidsWordList  = [];
let kidsWordIndex = 0;
let kidsFlipBusy  = false;   // prevent double-fire while model is running

const kidsWordPicker = $("kids-word-picker");
const kidsWordReader = $("kids-word-reader");
const kidsFlipInner  = $("kids-flip-inner");
const kidsFlipArea   = $("kids-flip-area");

function openKidsWordCategory(cat) {
  const words = cat === "mix" ? kidsWordBank : kidsWordBank.filter((w) => w.cat === cat);
  if (!words.length) return;
  kidsWordList  = words;
  kidsWordIndex = 0;

  const meta = cat === "mix"
    ? { label: "All Words", emoji: "🌍" }
    : (CAT_META[cat] || { label: cat, emoji: "📦" });
  $("kids-word-cat-title").textContent = `${meta.emoji} ${meta.label}`;

  kidsWordPicker.hidden = true;
  kidsWordReader.hidden = false;
  renderKidsWord();
}

function renderKidsWord() {
  const word  = kidsWordList[kidsWordIndex];
  const total = kidsWordList.length;

  // Reset flip state and clear any previous model output
  kidsFlipInner.classList.remove("is-flipped");
  kidsFlipBusy = false;

  $("kids-word-emoji-front").textContent = word.emoji;
  $("kids-word-emoji-back").textContent  = word.emoji;
  $("kids-word-am").textContent          = word.am;

  // Reset back side to loading state for fresh flip
  const omEl = $("kids-word-om");
  omEl.textContent = "";
  omEl.removeAttribute("data-loaded");

  const modelLabelEl = $("kids-word-model-label");
  if (modelLabelEl) modelLabelEl.textContent = "Model says:";

  const refEl = $("kids-word-reference");
  if (refEl) { refEl.hidden = true; refEl.textContent = ""; }

  $("kids-word-counter").textContent = `${kidsWordIndex + 1} / ${total}`;
  $("kids-word-prev").disabled = kidsWordIndex === 0;
  $("kids-word-next").disabled = kidsWordIndex === total - 1;

  // Dots
  const dots = $("kids-word-dots");
  dots.innerHTML = "";
  const maxDots = Math.min(total, 15);
  for (let i = 0; i < maxDots; i++) {
    const dot = document.createElement("span");
    dot.className = "kids-dot";
    if (total <= 15) {
      if (i === kidsWordIndex) dot.classList.add("is-current");
      else if (i < kidsWordIndex) dot.classList.add("is-done");
    } else {
      const cur = Math.round((kidsWordIndex / (total - 1)) * (maxDots - 1));
      if (i === cur) dot.classList.add("is-current");
      else if (i < cur) dot.classList.add("is-done");
    }
    dots.appendChild(dot);
  }
}

async function flipKidsCard() {
  const isFlipped = kidsFlipInner.classList.contains("is-flipped");

  if (!isFlipped) {
    // Flip to back — run model if not already done for this card
    const omEl = $("kids-word-om");
    if (!omEl.dataset.loaded) {
      if (kidsFlipBusy) return;
      kidsFlipBusy = true;

      const word = kidsWordList[kidsWordIndex];
      // Show loading state then flip
      omEl.textContent = "…";
      kidsFlipInner.classList.add("is-flipped");

      const result = await modelTranslate(word.am, "am2om");
      kidsFlipBusy = false;

      if (result) {
        omEl.setAttribute("lang", "om");
        omEl.textContent = result;
        omEl.dataset.loaded = "1";

        // Show reference for comparison
        const refEl = $("kids-word-reference");
        if (refEl) {
          refEl.textContent = `Reference: ${word.om}`;
          refEl.hidden = false;
        }
      } else {
        omEl.textContent = "(could not translate)";
      }
    } else {
      kidsFlipInner.classList.add("is-flipped");
    }
  } else {
    kidsFlipInner.classList.remove("is-flipped");
  }
}

kidsFlipArea.addEventListener("click", flipKidsCard);
kidsFlipArea.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); flipKidsCard(); }
});

$("kids-word-prev").addEventListener("click", () => {
  if (kidsWordIndex > 0) { kidsWordIndex--; renderKidsWord(); }
});
$("kids-word-next").addEventListener("click", () => {
  if (kidsWordIndex < kidsWordList.length - 1) { kidsWordIndex++; renderKidsWord(); }
});
$("kids-word-back").addEventListener("click", () => {
  kidsWordReader.hidden = true;
  kidsWordPicker.hidden = false;
});

// ---------------------------------------------------------------------------
// Kids — Word Quiz  (+ model confirmation after answering)
// ---------------------------------------------------------------------------

let kidsQuizQuestions = [], kidsQuizIndex = 0, kidsQuizCorrect = 0;
let kidsQuizCount = 4;

const kidsQuizSetup      = $("kids-quiz-setup");
const kidsQuizRunner     = $("kids-quiz-runner");
const kidsQuizResult     = $("kids-quiz-result");
const kidsStarsRow       = $("kids-stars-row");
const kidsQuizFeedback   = $("kids-quiz-feedback");
const kidsFeedbackEmoji  = $("kids-feedback-emoji");
const kidsFeedbackText   = $("kids-feedback-text");
const kidsQuizNextBtn    = $("kids-quiz-next-btn");
const kidsResultStars    = $("kids-result-stars");
const kidsResultMsg      = $("kids-result-msg");
const kidsResultScore    = $("kids-result-score");
const kidsQuizStartBtn   = $("kids-quiz-start-btn");
const kidsQuizRestartBtn = $("kids-quiz-restart-btn");
const kidsQuizModelBox   = $("kids-quiz-model-box");

document.querySelectorAll(".kids-count-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".kids-count-btn").forEach((b) => b.classList.remove("is-active"));
    btn.classList.add("is-active");
    kidsQuizCount = parseInt(btn.dataset.count, 10);
  });
});

const KIDS_CORRECT = [
  "Amazing! You got it! 🎉", "Yes! That's right! ⭐", "Wonderful! 🌟",
  "You're a star! 💫", "Perfect! 🏆", "Great job! 👏",
];
const KIDS_WRONG = [
  "Not quite — look at the green one! 🌿",
  "Almost! The right answer is glowing green. 💡",
  "That's okay! Check the green answer. 🌱",
];

kidsQuizStartBtn.addEventListener("click", async () => {
  if (!kidsDataLoaded) await loadKidsData();
  const selectedTile = $("kids-quiz-cat-grid").querySelector(".kids-cat-tile.is-selected");
  const cat = selectedTile ? selectedTile.dataset.cat : "mix";

  kidsQuizStartBtn.disabled = true;
  try {
    const res  = await fetch(`/api/words/quiz?category=${cat}&count=${kidsQuizCount}&direction=mix`);
    const data = await res.json();
    if (!res.ok) { alert(data.error || "Could not build a quiz."); return; }
    kidsQuizQuestions = data.questions;
    kidsQuizIndex = 0; kidsQuizCorrect = 0;
    kidsQuizSetup.hidden  = true;
    kidsQuizResult.hidden = true;
    kidsQuizRunner.hidden = false;
    renderKidsQuizQ();
  } finally { kidsQuizStartBtn.disabled = false; }
});

function renderKidsStars() {
  kidsStarsRow.innerHTML = "";
  kidsQuizQuestions.forEach((q, i) => {
    const s = document.createElement("span");
    if (i < kidsQuizIndex)        s.textContent = q._correct ? "⭐" : "💧";
    else if (i === kidsQuizIndex) s.textContent = "👉";
    else { s.textContent = "○"; s.style.opacity = "0.4"; }
    kidsStarsRow.appendChild(s);
  });
}

function renderKidsQuizQ() {
  const q          = kidsQuizQuestions[kidsQuizIndex];
  const promptLang = q.direction === "am2om" ? "am" : "om";
  const tgtLang    = q.direction === "am2om" ? "om" : "am";

  $("kids-quiz-direction").textContent =
    q.direction === "am2om"
      ? "🔵 What is this in Afan Oromo?"
      : "🟠 What is this in Amharic?";

  $("kids-quiz-emoji").textContent = q.emoji;
  $("kids-quiz-prompt").setAttribute("lang", promptLang);
  $("kids-quiz-prompt").textContent = q.prompt;

  $("kids-quiz-options").innerHTML = "";
  kidsQuizFeedback.hidden  = true;
  kidsQuizNextBtn.hidden   = true;

  // Hide model box — only shown after user answers
  if (kidsQuizModelBox) {
    kidsQuizModelBox.hidden = true;
    kidsQuizModelBox.innerHTML = "";
  }

  // Show a small loading indicator in the options area while model runs
  const loadingLi = document.createElement("li");
  loadingLi.id = "kids-options-loading";
  loadingLi.innerHTML = `<span class="options-loading-text">⏳ Loading options…</span>`;
  $("kids-quiz-options").appendChild(loadingLi);

  renderKidsStars();

  // Run model — its output becomes the correct answer
  modelTranslate(q.prompt, q.direction).then((modelOutput) => {
    // Remove loading indicator
    const loadingEl = document.getElementById("kids-options-loading");
    if (loadingEl) loadingEl.remove();

    let correctWord;
    if (modelOutput) {
      q._modelCorrect = modelOutput;
      correctWord     = modelOutput;
    } else {
      // Fallback to stored correct option
      q._modelCorrect = q.options[q.correct_index];
      correctWord     = q._modelCorrect;
    }

    // Build options: model answer + 3 distractors from word bank (same cat, different word)
    const pool        = q.cat === "mix" ? kidsWordBank : kidsWordBank.filter((w) => w.cat === q.cat);
    const distractors = pool
      .map((w) => w[tgtLang === "om" ? "om" : "am"])
      .filter((w) => w !== correctWord)
      .sort(() => Math.random() - 0.5)
      .slice(0, 3);

    const opts = [...distractors, correctWord];
    for (let i = opts.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [opts[i], opts[j]] = [opts[j], opts[i]];
    }
    q._opts       = opts;
    q._correctIdx = opts.indexOf(correctWord);

    // Render options
    opts.forEach((word, i) => {
      const li  = document.createElement("li");
      const btn = document.createElement("button");
      btn.className = "kids-quiz-option-btn";
      btn.setAttribute("lang", tgtLang);
      btn.textContent = word;
      btn.addEventListener("click", () => answerKidsQuiz(i, btn));
      li.appendChild(btn);
      $("kids-quiz-options").appendChild(li);
    });
  });
}

function answerKidsQuiz(chosenIndex, chosenBtn) {
  const q       = kidsQuizQuestions[kidsQuizIndex];
  const tgtLang = q.direction === "am2om" ? "om" : "am";
  const buttons = $("kids-quiz-options").querySelectorAll(".kids-quiz-option-btn");
  buttons.forEach((b) => (b.disabled = true));

  const isCorrect = chosenIndex === q._correctIdx;
  if (isCorrect) kidsQuizCorrect++;
  q._correct = isCorrect;

  chosenBtn.classList.add(isCorrect ? "is-correct" : "is-wrong");
  if (!isCorrect) buttons[q._correctIdx].classList.add("is-correct");

  kidsFeedbackEmoji.textContent = isCorrect ? "🎉" : "💡";
  kidsFeedbackText.textContent  = isCorrect
    ? KIDS_CORRECT[Math.floor(Math.random() * KIDS_CORRECT.length)]
    : KIDS_WRONG[Math.floor(Math.random() * KIDS_WRONG.length)];

  kidsQuizFeedback.hidden = false;
  kidsQuizNextBtn.hidden  = false;

  // Now reveal the model's answer
  if (kidsQuizModelBox && q._modelCorrect) {
    kidsQuizModelBox.hidden = false;
    kidsQuizModelBox.innerHTML =
      `<span class="kids-model-label">Model says:</span>
       <span class="kids-model-result" lang="${tgtLang}">${q._modelCorrect}</span>`;
  }

  renderKidsStars();
}

kidsQuizNextBtn.addEventListener("click", () => {
  kidsQuizIndex++;
  if (kidsQuizIndex >= kidsQuizQuestions.length) finishKidsQuiz();
  else renderKidsQuizQ();
});

function finishKidsQuiz() {
  kidsQuizRunner.hidden = true;
  kidsQuizResult.hidden = false;
  const pct   = kidsQuizCorrect / kidsQuizQuestions.length;
  const stars = pct === 1 ? 3 : pct >= 0.6 ? 2 : pct >= 0.3 ? 1 : 0;
  kidsResultStars.textContent = "⭐".repeat(stars) + "☆".repeat(3 - stars);
  const msgs = [
    "Keep practicing — you'll get it! 💪",
    "Good start! Try again for more stars! 🌱",
    "Great job! Almost perfect! 🎊",
    "PERFECT SCORE! You're incredible! 🏆🎉",
  ];
  kidsResultMsg.textContent   = msgs[stars];
  kidsResultScore.textContent = `${kidsQuizCorrect} out of ${kidsQuizQuestions.length} correct`;
}

kidsQuizRestartBtn.addEventListener("click", () => {
  kidsQuizResult.hidden = true;
  kidsQuizSetup.hidden  = false;
});
