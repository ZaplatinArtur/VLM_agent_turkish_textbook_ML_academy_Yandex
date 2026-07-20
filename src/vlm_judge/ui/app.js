const state = {
  tasks: [],
  goldTasks: [],
  adjudicationTasks: [],
  filtered: [],
  annotations: new Map(),
  gold: new Map(),
  adjudications: new Map(),
  mode: "judge",
  goldDirty: false,
  adjudicationDirty: false,
  adjudicationDecision: null,
  index: 0,
  score: null,
  winner: null,
  errors: new Set(),
  blind: true,
  autosaveTimer: null,
};

const $ = (id) => document.getElementById(id);
const els = {
  datasetName: $("datasetName"), progressText: $("progressText"), progressBar: $("progressBar"), savedState: $("savedState"),
  subjectFilter: $("subjectFilter"), typeFilter: $("typeFilter"), statusFilter: $("statusFilter"), searchInput: $("searchInput"), blindToggle: $("blindToggle"), workspaceMode: $("workspaceMode"), exportCsv: $("exportCsv"), exportJsonl: $("exportJsonl"),
  workspace: $("workspace"), emptyState: $("emptyState"), taskMeta: $("taskMeta"), taskTitle: $("taskTitle"),
  questionImage: $("questionImage"), questionImageButton: $("questionImageButton"), questionText: $("questionText"),
  referenceImage: $("referenceImage"), referenceImageButton: $("referenceImageButton"), referenceText: $("referenceText"), candidateText: $("candidateText"), candidateSetup: $("candidateSetup"),
  pointwiseComparison: $("pointwiseComparison"), pairwiseComparison: $("pairwiseComparison"), candidateAText: $("candidateAText"), candidateBText: $("candidateBText"), candidateASetup: $("candidateASetup"), candidateBSetup: $("candidateBSetup"),
  pairwiseGold: $("pairwiseGold"), pairwiseReferenceImage: $("pairwiseReferenceImage"), pairwiseReferenceImageButton: $("pairwiseReferenceImageButton"), pairwiseReferenceText: $("pairwiseReferenceText"),
  goldWorkspace: $("goldWorkspace"), goldReferenceImage: $("goldReferenceImage"), goldReferenceImageButton: $("goldReferenceImageButton"), goldReferenceText: $("goldReferenceText"), goldForm: $("goldForm"),
  goldTranscription: $("goldTranscription"), goldAcceptable: $("goldAcceptable"), goldSubanswers: $("goldSubanswers"), goldQuality: $("goldQuality"), goldNotes: $("goldNotes"), goldAnnotator: $("goldAnnotator"), goldSkipButton: $("goldSkipButton"), goldDraftButton: $("goldDraftButton"),
  adjudicationModeOption: $("adjudicationModeOption"), adjudicationWorkspace: $("adjudicationWorkspace"), adjudicationPriority: $("adjudicationPriority"), adjudicationReasons: $("adjudicationReasons"),
  humanReviewerMeta: $("humanReviewerMeta"), humanReviewScore: $("humanReviewScore"), humanReviewLabel: $("humanReviewLabel"), humanReviewConfidence: $("humanReviewConfidence"), humanReviewStrict: $("humanReviewStrict"), humanReviewRationale: $("humanReviewRationale"), humanReviewErrors: $("humanReviewErrors"),
  judgeReviewerMeta: $("judgeReviewerMeta"), judgeReviewScore: $("judgeReviewScore"), judgeReviewLabel: $("judgeReviewLabel"), judgeReviewConfidence: $("judgeReviewConfidence"), judgeReviewStrict: $("judgeReviewStrict"), judgeReviewRationale: $("judgeReviewRationale"), judgeReviewErrors: $("judgeReviewErrors"),
  adjudicationForm: $("adjudicationForm"), adjudicationDecisionButtons: $("adjudicationDecisionButtons"), adjudicationFinalScore: $("adjudicationFinalScore"), adjudicationIssueSource: $("adjudicationIssueSource"), adjudicationRationale: $("adjudicationRationale"), adjudicator: $("adjudicator"), adjudicationSkipButton: $("adjudicationSkipButton"), adjudicationDraftButton: $("adjudicationDraftButton"),
  pointwiseControls: $("pointwiseControls"), pairwiseControls: $("pairwiseControls"), scoreButtons: $("scoreButtons"), winnerButtons: $("winnerButtons"), errorTags: $("errorTags"),
  finalCorrect: $("finalCorrect"), reasoningCorrect: $("reasoningCorrect"), complete: $("complete"), referenceIssue: $("referenceIssue"), rationale: $("rationale"), confidence: $("confidence"), confidenceValue: $("confidenceValue"), annotator: $("annotator"),
  annotationForm: $("annotationForm"), prevButton: $("prevButton"), nextButton: $("nextButton"), skipButton: $("skipButton"),
  imageDialog: $("imageDialog"), dialogImage: $("dialogImage"), closeDialog: $("closeDialog"), toast: $("toast"),
};

function proxyImage(url) {
  return url ? `/api/image?url=${encodeURIComponent(url)}` : "";
}

function taskMode(task) {
  return task.candidate_a != null && task.candidate_b != null ? "pairwise" : "pointwise";
}

function annotationId(task) {
  if (task.annotation_id || task.pair_id) return task.annotation_id || task.pair_id;
  const setup = task.setup || "unknown";
  return `${task.task_id}::${setup}`;
}

function currentTask() {
  return state.filtered[state.index] || null;
}

function annotationFor(task) {
  return state.annotations.get(annotationId(task)) || {};
}

function goldFor(task) {
  return state.gold.get(task.task_id) || {};
}

function adjudicationFor(task) {
  return state.adjudications.get(task._adjudication?.adjudication_id) || {};
}

function sourceTasks() {
  if (state.mode === "gold") return state.goldTasks;
  if (state.mode === "adjudication") return state.adjudicationTasks;
  return state.tasks;
}

function currentRecordKey(task) {
  if (state.mode === "gold") return task.task_id;
  if (state.mode === "adjudication") return task._adjudication?.adjudication_id || annotationId(task);
  return annotationId(task);
}

function recordFor(task) {
  if (state.mode === "gold") return goldFor(task);
  if (state.mode === "adjudication") return adjudicationFor(task);
  return annotationFor(task);
}

function triValue(value) {
  if (value === true) return "true";
  if (value === false) return "false";
  if (value === null) return "null";
  return "";
}

function parseTri(value) {
  if (value === "true") return true;
  if (value === "false") return false;
  if (value === "null") return null;
  return undefined;
}

function labelForScore(score) {
  return ["incorrect", "partially_correct", "partially_correct", "mostly_correct", "fully_correct"][score] || null;
}

function difficulty(task) {
  const meta = task.metadata || {};
  return ["easy", "medium", "hard"].find((key) => meta[key]) || "unlabeled";
}

function setImage(image, button, url) {
  if (!url) {
    button.hidden = true;
    image.removeAttribute("src");
    return;
  }
  button.hidden = false;
  image.src = proxyImage(url);
  image.dataset.original = proxyImage(url);
  image.onerror = () => { button.hidden = true; };
}

function renderFilters() {
  const tasks = sourceTasks();
  const subjects = [...new Set(tasks.map((task) => task.subject || "unknown"))].sort();
  const types = [...new Set(tasks.map((task) => task.answer_type || "unknown"))].sort();
  els.subjectFilter.replaceChildren(new Option("Все", ""));
  els.typeFilter.replaceChildren(new Option("Все", ""));
  for (const subject of subjects) els.subjectFilter.add(new Option(subject, subject));
  for (const type of types) els.typeFilter.add(new Option(type, type));
}

function applyFilters({ preserveTaskId = true } = {}) {
  const activeId = preserveTaskId && currentTask() ? currentRecordKey(currentTask()) : null;
  const subject = els.subjectFilter.value;
  const type = els.typeFilter.value;
  const status = els.statusFilter.value;
  const query = els.searchInput.value.trim().toLowerCase();
  const requestedStatus = status === "complete"
    ? (state.mode === "gold" ? "verified" : state.mode === "adjudication" ? "resolved" : "complete")
    : status;
  state.filtered = sourceTasks().filter((task) => {
    const record = recordFor(task);
    const taskStatus = record?.status || "unlabeled";
    const haystack = [task.task_id, task.subject, task.metadata?.topic_area, task.metadata?.sub_topic].filter(Boolean).join(" ").toLowerCase();
    return (!subject || task.subject === subject)
      && (!type || task.answer_type === type)
      && (!requestedStatus || taskStatus === requestedStatus)
      && (!query || haystack.includes(query));
  });
  const preservedIndex = activeId ? state.filtered.findIndex((task) => currentRecordKey(task) === activeId) : -1;
  state.index = preservedIndex >= 0 ? preservedIndex : Math.min(state.index, Math.max(0, state.filtered.length - 1));
  render();
}

function renderMeta(task) {
  const values = [
    task.subject || "unknown",
    task.grade != null ? `класс ${task.grade}` : null,
    task.answer_type || "unknown",
    difficulty(task),
    task.task_id,
  ].filter(Boolean);
  els.taskMeta.innerHTML = values.map((value) => `<span class="meta-pill">${escapeHtml(String(value))}</span>`).join("");
}

function escapeHtml(value) {
  return value.replace(/[&<>"]/g, (character) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[character]));
}

function setSelected(button, selected) {
  button.classList.toggle("selected", selected);
  button.setAttribute("aria-pressed", String(selected));
}

function render() {
  const task = currentTask();
  const tasks = sourceTasks();
  const doneStatus = state.mode === "gold" ? "verified" : state.mode === "adjudication" ? "resolved" : "complete";
  const completed = tasks.filter((value) => recordFor(value).status === doneStatus).length;
  els.progressText.textContent = `${completed} / ${tasks.length}`;
  els.progressBar.style.width = `${tasks.length ? completed / tasks.length * 100 : 0}%`;
  els.workspace.hidden = !task;
  els.emptyState.hidden = Boolean(task);
  if (!task) return;

  renderMeta(task);
  els.taskTitle.textContent = task.metadata?.topic_area || task.metadata?.synthetic_case || `Задание ${task.task_id}`;
  setImage(els.questionImage, els.questionImageButton, task.question_image_url);
  els.questionText.textContent = task.question_text || "";
  els.questionText.hidden = !task.question_text;

  const goldMode = state.mode === "gold";
  const adjudicationMode = state.mode === "adjudication";
  els.goldWorkspace.hidden = !goldMode;
  els.adjudicationWorkspace.hidden = !adjudicationMode;
  els.annotationForm.hidden = goldMode || adjudicationMode;
  els.blindToggle.closest("label").hidden = goldMode || adjudicationMode;
  if (goldMode) {
    els.pointwiseComparison.hidden = true;
    els.pairwiseComparison.hidden = true;
    els.pairwiseGold.hidden = true;
    setImage(els.goldReferenceImage, els.goldReferenceImageButton, task.reference_image_url);
    els.goldReferenceText.textContent = task.reference_answer || "";
    loadGold(task);
    els.prevButton.disabled = state.index === 0;
    els.nextButton.disabled = state.index >= state.filtered.length - 1;
    return;
  }

  setImage(els.referenceImage, els.referenceImageButton, task.reference_image_url);
  els.referenceText.textContent = task.reference_answer || "";

  if (!adjudicationMode) {
    const primaryLabelLocked = annotationFor(task).status !== "complete";
    els.blindToggle.disabled = primaryLabelLocked;
    if (primaryLabelLocked) {
      els.blindToggle.checked = true;
      state.blind = true;
    }
  } else {
    els.blindToggle.checked = true;
    els.blindToggle.disabled = true;
    state.blind = true;
  }

  const mode = taskMode(task);
  const pointwise = mode === "pointwise";
  els.pointwiseComparison.hidden = !pointwise;
  els.pointwiseControls.hidden = !pointwise || adjudicationMode;
  els.pairwiseComparison.hidden = pointwise || adjudicationMode;
  els.pairwiseControls.hidden = pointwise || adjudicationMode;
  els.pairwiseGold.hidden = pointwise || adjudicationMode;
  if (pointwise) {
    els.candidateText.textContent = task.candidate_answer || "Ответ агента ещё не прикреплён.";
    els.candidateSetup.textContent = state.blind ? "setup скрыт" : (task.setup || "setup неизвестен");
  } else {
    els.candidateAText.textContent = task.candidate_a || "";
    els.candidateBText.textContent = task.candidate_b || "";
    els.candidateASetup.textContent = state.blind ? "setup скрыт" : (task.metadata?.candidate_a_setup || "setup неизвестен");
    els.candidateBSetup.textContent = state.blind ? "setup скрыт" : (task.metadata?.candidate_b_setup || "setup неизвестен");
    setImage(els.pairwiseReferenceImage, els.pairwiseReferenceImageButton, task.reference_image_url);
    els.pairwiseReferenceText.textContent = task.reference_answer || "";
  }

  if (adjudicationMode) loadAdjudication(task);
  else loadAnnotation(task);
  els.prevButton.disabled = state.index === 0;
  els.nextButton.disabled = state.index >= state.filtered.length - 1;
}

function loadGold(task) {
  const record = goldFor(task);
  els.goldTranscription.value = record.transcription ?? task.reference_answer ?? "";
  els.goldAcceptable.value = (record.acceptable_answers || []).join("\n");
  els.goldSubanswers.value = (record.subanswers || []).join("\n");
  els.goldQuality.value = record.quality || "unknown";
  els.goldNotes.value = record.notes || "";
  els.goldAnnotator.value = record.annotator || localStorage.getItem("vlm-annotator") || "";
  state.goldDirty = false;
  els.savedState.textContent = record.status === "verified" ? "проверено" : record.status === "draft" ? "черновик" : record.status === "skipped" ? "пропущено" : "не сохранено";
}

function loadAnnotation(task) {
  const annotation = annotationFor(task);
  state.score = Number.isInteger(annotation.score) ? annotation.score : null;
  state.winner = annotation.winner || null;
  state.errors = new Set(annotation.error_types || []);
  [...els.scoreButtons.querySelectorAll("button")].forEach((button) => setSelected(button, Number(button.dataset.score) === state.score));
  [...els.winnerButtons.querySelectorAll("button")].forEach((button) => setSelected(button, button.dataset.winner === state.winner));
  [...els.errorTags.querySelectorAll("button")].forEach((button) => setSelected(button, state.errors.has(button.dataset.error)));
  els.finalCorrect.value = triValue(annotation.final_answer_correct);
  els.reasoningCorrect.value = triValue(annotation.reasoning_correct);
  els.complete.value = triValue(annotation.complete);
  els.referenceIssue.checked = Boolean(annotation.reference_quality_issue);
  els.rationale.value = annotation.rationale || "";
  els.confidence.value = Math.round((annotation.confidence ?? 0.8) * 100);
  els.confidenceValue.textContent = `${els.confidence.value}%`;
  els.annotator.value = annotation.annotator || localStorage.getItem("vlm-annotator") || "";
  els.savedState.textContent = annotation.status === "complete" ? "готово" : annotation.status === "draft" ? "черновик" : annotation.status === "skipped" ? "пропущено" : "не сохранено";
}

const adjudicationReasonLabels = {
  judge_error: "ошибка LLM-судьи",
  score_disagreement: "разные оценки 0–4",
  strict_disagreement: "расхождение exact match",
  low_judge_confidence: "низкая уверенность LLM",
  reference_quality_issue: "проблема эталона",
  agreement_control: "контрольная выборка согласий",
};

function displayBoolean(value) {
  return value === true ? "да" : value === false ? "нет" : "—";
}

function displayConfidence(value) {
  return typeof value === "number" ? `${Math.round(value * 100)}%` : "—";
}

function displayScore(value) {
  return Number.isInteger(value) ? String(value) : "—";
}

function renderReviewErrors(container, errors) {
  const values = Array.isArray(errors) ? errors.filter(Boolean) : [];
  container.innerHTML = values.map((value) => `<span>${escapeHtml(String(value))}</span>`).join("");
}

function loadAdjudication(task) {
  const context = task._adjudication || {};
  const human = context.human || {};
  const judgeResult = context.judge_result || {};
  const verdict = judgeResult.verdict || {};
  const judgeMeta = judgeResult.judge || {};
  const existing = adjudicationFor(task);

  els.adjudicationPriority.textContent = `priority ${context.priority ?? 0}`;
  els.adjudicationReasons.innerHTML = (context.reasons || [])
    .map((reason) => `<span class="meta-pill">${escapeHtml(adjudicationReasonLabels[reason] || reason)}</span>`)
    .join("");

  els.humanReviewerMeta.textContent = human.annotator || "анонимный разметчик";
  els.humanReviewScore.textContent = displayScore(human.score);
  els.humanReviewLabel.textContent = human.label || (human.score === 4 ? "fully_correct" : "human verdict");
  els.humanReviewConfidence.textContent = displayConfidence(human.confidence);
  els.humanReviewStrict.textContent = displayBoolean(
    typeof human.strict_correct === "boolean" ? human.strict_correct : (Number.isInteger(human.score) ? human.score === 4 : null)
  );
  els.humanReviewRationale.textContent = human.rationale || "";
  renderReviewErrors(els.humanReviewErrors, human.error_types);

  els.judgeReviewerMeta.textContent = [judgeMeta.backend, judgeMeta.model].filter(Boolean).join(" · ") || "неизвестный backend";
  els.judgeReviewScore.textContent = displayScore(verdict.score);
  els.judgeReviewLabel.textContent = verdict.label || (judgeMeta.error ? "ошибка judge" : "нет вердикта");
  els.judgeReviewConfidence.textContent = displayConfidence(verdict.confidence);
  els.judgeReviewStrict.textContent = displayBoolean(verdict.strict_correct);
  els.judgeReviewRationale.textContent = verdict.rationale || judgeMeta.error || "";
  renderReviewErrors(els.judgeReviewErrors, verdict.error_types || (judgeMeta.error ? [judgeMeta.error] : []));

  state.adjudicationDecision = existing.decision || null;
  els.adjudicationFinalScore.value = Number.isInteger(existing.final_score) ? String(existing.final_score) : "";
  els.adjudicationIssueSource.value = existing.issue_source || "none";
  els.adjudicationRationale.value = existing.rationale || "";
  els.adjudicator.value = existing.adjudicator || localStorage.getItem("vlm-annotator") || "";
  state.adjudicationDirty = false;
  renderAdjudicationDecisionSelection();
  els.savedState.textContent = existing.status === "resolved" ? "разрешено" : existing.status === "draft" ? "черновик" : existing.status === "skipped" ? "пропущено" : "не сохранено";
}

function collectAnnotation(status) {
  const task = currentTask();
  const mode = taskMode(task);
  const annotation = {
    annotation_id: annotationId(task), task_id: task.task_id, mode, status,
    setup: mode === "pointwise" ? (task.setup || "unknown") : null,
    pair_id: mode === "pairwise" ? (task.pair_id || annotationId(task)) : null,
    subject: task.subject || "unknown",
    grade: task.grade ?? null,
    answer_type: task.answer_type || "unknown",
    candidate_a_setup: mode === "pairwise" ? (task.metadata?.candidate_a_setup || null) : null,
    candidate_b_setup: mode === "pairwise" ? (task.metadata?.candidate_b_setup || null) : null,
    side_swapped: mode === "pairwise" ? Boolean(task.metadata?.side_swapped) : null,
    mirrored: mode === "pairwise" ? Boolean(task.metadata?.mirrored) : null,
    score: mode === "pointwise" ? state.score : null,
    label: mode === "pointwise" && state.score != null ? labelForScore(state.score) : null,
    strict_correct: mode === "pointwise" ? state.score === 4 : null,
    final_answer_correct: parseTri(els.finalCorrect.value),
    reasoning_correct: parseTri(els.reasoningCorrect.value),
    complete: parseTri(els.complete.value),
    winner: mode === "pairwise" ? state.winner : null,
    confidence: Number(els.confidence.value) / 100,
    reference_quality_issue: els.referenceIssue.checked,
    error_types: [...state.errors].sort(),
    rationale: els.rationale.value.trim(),
    annotator: els.annotator.value.trim(),
  };
  localStorage.setItem("vlm-annotator", annotation.annotator);
  return annotation;
}

async function save(status = "draft", { quiet = false } = {}) {
  clearTimeout(state.autosaveTimer);
  state.autosaveTimer = null;
  const task = currentTask();
  if (!task) return false;
  if (status === "complete" && taskMode(task) === "pointwise" && state.score == null) {
    showToast("Сначала выберите оценку 0–4");
    return false;
  }
  if (status === "complete" && taskMode(task) === "pairwise" && !state.winner) {
    showToast("Сначала выберите победителя");
    return false;
  }
  const annotation = collectAnnotation(status);
  els.savedState.textContent = "сохранение…";
  const response = await fetch("/api/annotations", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(annotation)});
  if (!response.ok) {
    const message = await response.text();
    els.savedState.textContent = "ошибка";
    if (!quiet) showToast(`Ошибка сохранения: ${message}`);
    return false;
  }
  const payload = await response.json();
  state.annotations.set(annotation.annotation_id, payload.saved);
  els.savedState.textContent = status === "complete" ? "готово" : status === "skipped" ? "пропущено" : "черновик";
  if (!quiet) showToast("Сохранено");
  return true;
}

function nonEmptyLines(value) {
  return value.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
}

function collectGold(status) {
  const task = currentTask();
  const record = {
    task_id: task.task_id,
    status,
    subject: task.subject || "unknown",
    grade: task.grade ?? null,
    answer_type: task.answer_type || "unknown",
    transcription: els.goldTranscription.value.trim(),
    acceptable_answers: nonEmptyLines(els.goldAcceptable.value),
    subanswers: nonEmptyLines(els.goldSubanswers.value),
    quality: els.goldQuality.value,
    notes: els.goldNotes.value.trim(),
    annotator: els.goldAnnotator.value.trim(),
    source_reference_answer: task.reference_answer || null,
    source_reference_image_url: task.reference_image_url || null,
  };
  localStorage.setItem("vlm-annotator", record.annotator);
  return record;
}

async function saveGold(status = "draft", {quiet = false} = {}) {
  const task = currentTask();
  if (!task) return false;
  const record = collectGold(status);
  const noTranscriptionAllowed = ["incorrect", "unreadable"].includes(record.quality);
  if (status === "verified" && !record.transcription && !record.subanswers.length && !noTranscriptionAllowed) {
    showToast("Добавьте транскрипцию или обязательные подответы");
    return false;
  }
  els.savedState.textContent = "сохранение…";
  const response = await fetch("/api/gold", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(record),
  });
  if (!response.ok) {
    els.savedState.textContent = "ошибка";
    showToast(`Ошибка сохранения: ${await response.text()}`);
    return false;
  }
  const payload = await response.json();
  state.gold.set(task.task_id, payload.saved);
  state.goldDirty = false;
  els.savedState.textContent = status === "verified" ? "проверено" : status === "skipped" ? "пропущено" : "черновик";
  if (!quiet) showToast("Gold сохранён");
  return true;
}

function collectAdjudication(status) {
  const task = currentTask();
  const context = task._adjudication || {};
  const human = context.human || {};
  const judgeVerdict = context.judge_result?.verdict || {};
  const scoreValue = els.adjudicationFinalScore.value;
  const finalScore = scoreValue === "" ? null : Number(scoreValue);
  const record = {
    adjudication_id: context.adjudication_id,
    task_id: task.task_id,
    setup: task.setup || "unknown",
    status,
    decision: state.adjudicationDecision,
    final_score: finalScore,
    final_label: finalScore == null ? null : labelForScore(finalScore),
    strict_correct: finalScore == null ? null : finalScore === 4,
    issue_source: els.adjudicationIssueSource.value,
    queue_reasons: context.reasons || [],
    queue_priority: context.priority ?? null,
    human_score: Number.isInteger(human.score) ? human.score : null,
    judge_score: Number.isInteger(judgeVerdict.score) ? judgeVerdict.score : null,
    judge_request_id: context.judge_result?.request_id || null,
    rationale: els.adjudicationRationale.value.trim(),
    adjudicator: els.adjudicator.value.trim(),
  };
  localStorage.setItem("vlm-annotator", record.adjudicator);
  return record;
}

async function saveAdjudication(status = "draft", {quiet = false} = {}) {
  const task = currentTask();
  if (!task) return false;
  const record = collectAdjudication(status);
  if (status === "resolved" && !record.decision) {
    showToast("Сначала выберите финальное решение");
    return false;
  }
  if (status === "resolved" && record.decision !== "exclude" && record.final_score == null) {
    showToast("Укажите финальную оценку 0–4");
    return false;
  }
  els.savedState.textContent = "сохранение…";
  const response = await fetch("/api/adjudications", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(record),
  });
  if (!response.ok) {
    els.savedState.textContent = "ошибка";
    if (!quiet) showToast(`Ошибка сохранения: ${await response.text()}`);
    return false;
  }
  const payload = await response.json();
  state.adjudications.set(record.adjudication_id, payload.saved);
  state.adjudicationDirty = false;
  els.savedState.textContent = status === "resolved" ? "разрешено" : status === "skipped" ? "пропущено" : "черновик";
  if (!quiet) showToast("Adjudication сохранён");
  return true;
}

function renderAdjudicationDecisionSelection() {
  [...els.adjudicationDecisionButtons.querySelectorAll("button[data-decision]")]
    .forEach((button) => setSelected(button, button.dataset.decision === state.adjudicationDecision));
}

function chooseAdjudicationDecision(decision) {
  const context = currentTask()?._adjudication || {};
  state.adjudicationDecision = decision;
  if (decision === "human" && Number.isInteger(context.human?.score)) {
    els.adjudicationFinalScore.value = String(context.human.score);
  } else if (decision === "judge" && Number.isInteger(context.judge_result?.verdict?.score)) {
    els.adjudicationFinalScore.value = String(context.judge_result.verdict.score);
  } else if (decision === "exclude") {
    els.adjudicationFinalScore.value = "";
    if ((context.reasons || []).includes("reference_quality_issue")) els.adjudicationIssueSource.value = "reference";
  }
  state.adjudicationDirty = true;
  els.savedState.textContent = "изменено";
  renderAdjudicationDecisionSelection();
}

function scheduleAutosave() {
  clearTimeout(state.autosaveTimer);
  els.savedState.textContent = "изменено";
  const scheduledId = currentTask() ? annotationId(currentTask()) : null;
  state.autosaveTimer = setTimeout(() => {
    state.autosaveTimer = null;
    if (currentTask() && annotationId(currentTask()) === scheduledId) save("draft", {quiet: true});
  }, 650);
}

async function move(delta) {
  if (state.mode === "judge" && state.autosaveTimer) await save("draft", {quiet: true});
  if (state.mode === "gold" && state.goldDirty) await saveGold("draft", {quiet: true});
  if (state.mode === "adjudication" && state.adjudicationDirty) await saveAdjudication("draft", {quiet: true});
  state.index = Math.max(0, Math.min(state.filtered.length - 1, state.index + delta));
  render();
  window.scrollTo({top: 0, behavior: "smooth"});
}

function showToast(message) {
  els.toast.textContent = message;
  els.toast.classList.add("show");
  setTimeout(() => els.toast.classList.remove("show"), 1800);
}

function openImage(image) {
  if (!image.dataset.original) return;
  els.dialogImage.src = image.dataset.original;
  els.imageDialog.showModal();
}

async function switchWorkspaceMode(mode) {
  if (state.mode === "judge" && state.autosaveTimer) await save("draft", {quiet: true});
  if (state.mode === "gold" && state.goldDirty) await saveGold("draft", {quiet: true});
  if (state.mode === "adjudication" && state.adjudicationDirty) await saveAdjudication("draft", {quiet: true});
  state.mode = mode;
  state.index = 0;
  els.exportCsv.href = mode === "gold" ? "/api/export-gold.csv" : mode === "adjudication" ? "/api/export-adjudications.csv" : "/api/export.csv";
  els.exportJsonl.href = mode === "gold" ? "/api/export-gold.jsonl" : mode === "adjudication" ? "/api/export-adjudications.jsonl" : "/api/export.jsonl";
  els.subjectFilter.value = "";
  els.typeFilter.value = "";
  els.statusFilter.value = "";
  els.searchInput.value = "";
  renderFilters();
  applyFilters({preserveTaskId: false});
}

function bindEvents() {
  [els.subjectFilter, els.typeFilter, els.statusFilter].forEach((element) => element.addEventListener("change", () => applyFilters()));
  els.searchInput.addEventListener("input", () => applyFilters());
  els.workspaceMode.addEventListener("change", () => switchWorkspaceMode(els.workspaceMode.value));
  els.blindToggle.addEventListener("change", () => { state.blind = els.blindToggle.checked; render(); });
  els.prevButton.addEventListener("click", () => move(-1));
  els.nextButton.addEventListener("click", () => move(1));
  els.scoreButtons.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-score]"); if (!button) return;
    state.score = Number(button.dataset.score); renderScoreSelection(); scheduleAutosave();
  });
  els.winnerButtons.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-winner]"); if (!button) return;
    state.winner = button.dataset.winner; renderWinnerSelection(); scheduleAutosave();
  });
  els.errorTags.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-error]"); if (!button) return;
    const error = button.dataset.error; state.errors.has(error) ? state.errors.delete(error) : state.errors.add(error);
    setSelected(button, state.errors.has(error)); scheduleAutosave();
  });
  [els.finalCorrect, els.reasoningCorrect, els.complete, els.referenceIssue, els.rationale, els.annotator].forEach((element) => element.addEventListener("input", scheduleAutosave));
  els.confidence.addEventListener("input", () => { els.confidenceValue.textContent = `${els.confidence.value}%`; scheduleAutosave(); });
  els.annotationForm.addEventListener("submit", async (event) => { event.preventDefault(); if (await save("complete")) advanceAfterSave(); });
  els.skipButton.addEventListener("click", async () => { if (await save("skipped")) advanceAfterSave(); });
  els.questionImageButton.addEventListener("click", () => openImage(els.questionImage));
  els.referenceImageButton.addEventListener("click", () => openImage(els.referenceImage));
  els.pairwiseReferenceImageButton.addEventListener("click", () => openImage(els.pairwiseReferenceImage));
  els.goldReferenceImageButton.addEventListener("click", () => openImage(els.goldReferenceImage));
  els.goldForm.addEventListener("submit", async (event) => { event.preventDefault(); if (await saveGold("verified")) advanceAfterGold(); });
  els.goldDraftButton.addEventListener("click", async () => { if (await saveGold("draft")) advanceAfterGold(); });
  els.goldSkipButton.addEventListener("click", async () => { if (await saveGold("skipped")) advanceAfterGold(); });
  [els.goldTranscription, els.goldAcceptable, els.goldSubanswers, els.goldQuality, els.goldNotes, els.goldAnnotator]
    .forEach((element) => element.addEventListener("input", () => { state.goldDirty = true; els.savedState.textContent = "изменено"; }));
  els.adjudicationDecisionButtons.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-decision]");
    if (button) chooseAdjudicationDecision(button.dataset.decision);
  });
  [els.adjudicationFinalScore, els.adjudicationIssueSource, els.adjudicationRationale, els.adjudicator]
    .forEach((element) => element.addEventListener("input", () => { state.adjudicationDirty = true; els.savedState.textContent = "изменено"; }));
  els.adjudicationForm.addEventListener("submit", async (event) => { event.preventDefault(); if (await saveAdjudication("resolved")) advanceAfterAdjudication(); });
  els.adjudicationDraftButton.addEventListener("click", async () => { if (await saveAdjudication("draft")) advanceAfterAdjudication(); });
  els.adjudicationSkipButton.addEventListener("click", async () => { if (await saveAdjudication("skipped")) advanceAfterAdjudication(); });
  els.closeDialog.addEventListener("click", () => els.imageDialog.close());
  els.imageDialog.addEventListener("click", (event) => { if (event.target === els.imageDialog) els.imageDialog.close(); });
  document.addEventListener("keydown", async (event) => {
    const editing = ["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName);
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
      event.preventDefault();
      if (state.mode === "gold") { if (await saveGold("verified")) advanceAfterGold(); }
      else if (state.mode === "adjudication") { if (await saveAdjudication("resolved")) advanceAfterAdjudication(); }
      else if (await save("complete")) advanceAfterSave();
      return;
    }
    if (editing) return;
    if (state.mode === "judge" && /^[0-4]$/.test(event.key) && currentTask() && taskMode(currentTask()) === "pointwise") { state.score = Number(event.key); renderScoreSelection(); scheduleAutosave(); }
    if (event.key === "ArrowLeft") move(-1);
    if (event.key === "ArrowRight") move(1);
  });
}

function advanceAfterSave() {
  clearTimeout(state.autosaveTimer);
  state.autosaveTimer = null;
  const previousId = currentTask() ? annotationId(currentTask()) : null;
  const previousIndex = state.index;
  applyFilters();
  if (!state.filtered.length) return;
  const stillVisible = state.filtered.findIndex((task) => annotationId(task) === previousId);
  state.index = stillVisible >= 0
    ? Math.min(stillVisible + 1, state.filtered.length - 1)
    : Math.min(previousIndex, state.filtered.length - 1);
  render();
  window.scrollTo({top: 0, behavior: "smooth"});
}

function advanceAfterGold() {
  const previousId = currentTask()?.task_id || null;
  const previousIndex = state.index;
  applyFilters();
  if (!state.filtered.length) return;
  const stillVisible = state.filtered.findIndex((task) => task.task_id === previousId);
  state.index = stillVisible >= 0
    ? Math.min(stillVisible + 1, state.filtered.length - 1)
    : Math.min(previousIndex, state.filtered.length - 1);
  render();
  window.scrollTo({top: 0, behavior: "smooth"});
}

function advanceAfterAdjudication() {
  const previousId = currentTask()?._adjudication?.adjudication_id || null;
  const previousIndex = state.index;
  applyFilters();
  if (!state.filtered.length) return;
  const stillVisible = state.filtered.findIndex((task) => task._adjudication?.adjudication_id === previousId);
  state.index = stillVisible >= 0
    ? Math.min(stillVisible + 1, state.filtered.length - 1)
    : Math.min(previousIndex, state.filtered.length - 1);
  render();
  window.scrollTo({top: 0, behavior: "smooth"});
}

function renderScoreSelection() {
  [...els.scoreButtons.querySelectorAll("button")].forEach((button) => setSelected(button, Number(button.dataset.score) === state.score));
}

function renderWinnerSelection() {
  [...els.winnerButtons.querySelectorAll("button")].forEach((button) => setSelected(button, button.dataset.winner === state.winner));
}

async function init() {
  bindEvents();
  const [tasksResponse, annotationsResponse, goldResponse, adjudicationContextResponse, adjudicationsResponse] = await Promise.all([
    fetch("/api/tasks"),
    fetch("/api/annotations"),
    fetch("/api/gold"),
    fetch("/api/adjudication-context"),
    fetch("/api/adjudications"),
  ]);
  if (![tasksResponse, annotationsResponse, goldResponse, adjudicationContextResponse, adjudicationsResponse].every((response) => response.ok)) throw new Error("Не удалось загрузить данные");
  const taskPayload = await tasksResponse.json();
  const annotationPayload = await annotationsResponse.json();
  const goldPayload = await goldResponse.json();
  const adjudicationPayload = await adjudicationContextResponse.json();
  const adjudicationsPayload = await adjudicationsResponse.json();
  state.tasks = taskPayload.tasks;
  const seenTaskIds = new Set();
  state.goldTasks = state.tasks.filter((task) => {
    if (seenTaskIds.has(task.task_id)) return false;
    seenTaskIds.add(task.task_id);
    return true;
  });
  state.annotations = new Map(annotationPayload.annotations.map((annotation) => [annotation.annotation_id || annotation.task_id, annotation]));
  state.gold = new Map(goldPayload.gold.map((record) => [record.task_id, record]));
  state.adjudicationTasks = adjudicationPayload.items || [];
  state.adjudications = new Map((adjudicationsPayload.adjudications || []).map((record) => [record.adjudication_id, record]));
  els.adjudicationModeOption.disabled = !adjudicationPayload.enabled || !state.adjudicationTasks.length;
  els.adjudicationModeOption.textContent = state.adjudicationTasks.length ? `Adjudication (${state.adjudicationTasks.length})` : "Adjudication — нет очереди";
  els.datasetName.textContent = taskPayload.dataset;
  renderFilters();
  applyFilters({preserveTaskId: false});
}

init().catch((error) => {
  els.emptyState.hidden = false;
  els.emptyState.innerHTML = `<h1>Ошибка запуска</h1><p>${escapeHtml(error.message)}</p>`;
});
