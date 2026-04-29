const socket = io();
let selectedMode = 'smart';
let selectedLang = 'uk';
let sessionStats = { questions: 0, tests: 0, symptoms: 0, ig: 0 };
let diagnosisNames = [];
let clientStep = 0;

// ── Language Labels ────────────────────────────────────────
const L = {
  uk: {
    // chat
    sessionStarted:    'Сесія розпочата. Пацієнт чекає. Задайте перше питання.',
    finalDiag:         (d) => `Ваш остаточний діагноз: "${d}". Генерація звіту…`,
    doctor:            'Лікар',
    patient:           'Пацієнт',
    lab:               'Лабораторія',
    system:            'Система',
    step:              (n) => `Крок ${n}`,
    question:          'Питання',
    test:              'Тест',
    revealed:          (name, pct) => `✅ Виявлено: <strong>${name}</strong> <span style="opacity:.6;font-size:10px">(${pct}%)</span>`,
    noHints:           'Немає доступних рекомендацій',
    male:              'Чоловік',
    female:            'Жінка',
    sex_m:             '♂ Пацієнт',
    sex_f:             '♀ Пацієнтка',
    symptoms:          'симптомів',
    loading:           'Завантаження…',
    // splash
    splashLangLabel:   '🌐 Мова спілкування з пацієнтом',
    splashInitStatus:  'Оберіть режим і натисніть «Почати»',
    splashInitializing:'Ініціалізація системи…',
    btnStart:          'Почати сесію',
    modeSmartDesc:     'LLM-генеровані реалістичні відповіді (потрібен Ollama/Gemma)',
    modeSimpleDesc:    'Векторна схожість без LLM. Швидко та стабільно.',
    // topbar
    tbStep:            'Крок',
    // left panel
    panelPatient:      'Пацієнт',
    panelSymptoms:     'Виявлені симптоми',
    complaintsLabel:   '📋 Скарги',
    entropyLabel:      'Невизначеність (Ентропія)',
    noSymptoms:        'Поки що нічого не виявлено',
    uncertaintyDown:   'невизначеність ↓',
    uncertaintyUp:     'невизначеність ↑',
    // input area
    inputPlaceholder:  'Задайте питання пацієнту… (Enter для відправки)',
    btnAsk:            '➤ Запитати',
    btnTest:           '🔬 Тест <span id="test-counter" class="test-counter">0/3</span>',
    chipChestPain:     'Біль у грудях',
    chipBreath:        'Задишка',
    chipFever:         'Гарячка',
    chipFatigue:       'Слабкість',
    chipMeds:          'Медикаменти',
    chipDiagnose:      '🩺 Поставити діагноз',
    // hints panel
    hintsTitle:        '💡 Підказки',
    hintsActive:       'Активно',
    hintsLockTitle:    'Підказки недоступні',
    hintsLockSub:      'Зробіть ще',
    hintsLockSub2:     'кроків самостійно',
    hintsIntro:        'Система рекомендує ці питання на основі поточного стану діагностики:',
    hintsRefresh:      '↻ Оновити підказки',
    hintsAnalyzing:    'Аналіз стану…',
    // diagnosis dialog
    diagTitle:         '🩺 Остаточний діагноз',
    diagSub:           'Введіть ваш діагноз для завершення сесії та отримання оцінки',
    diagPlaceholder:   'Наприклад: Spontaneous pneumothorax',
    btnCancel:         'Скасувати',
    btnConfirmDiag:    'Підтвердити діагноз',
    // report modal
    reportTitle:       'Звіт компетенцій',
    reportSubtitle:    'Результати діагностичної сесії',
    btnNewSession:     '🔄 Нова сесія',
    resultCorrect:     'ПРАВИЛЬНИЙ ДІАГНОЗ',
    resultWrong:       'НЕПРАВИЛЬНИЙ ДІАГНОЗ',
    resultYours:       'Ваш:',
    resultTrue:        'Справжній:',
    scoreTotal:        'Загальний бал',
    scoreComm:         'Комунікація',
    scoreIG:           'Загальний IG',
    scoreCritical:     'Критичних помилок',
    tableAction:       'Дія',
    tableScore:        'Бал',
    // test dialog
    testDialogTitle:   '🔬 Замовити аналіз / обстеження',
    testDialogSub:     (left, max) => `Залишилось тестів: <strong>${left}</strong> з ${max}`,
    testSearchPH:      'Пошук аналізу…',
    testSelected:      'Обрано:',
    btnConfirmTest:    'Провести аналіз',
    testNoResults:     'Нічого не знайдено',
    testSearchResults: (n) => `🔍 Результати пошуку (${n})`,
    testLimitAlert:    'Ліміт тестів вичерпано',
    testWaitAlert:     'Зачекайте завершення ініціалізації сесії.',
    // proc categories
    catBlood:          '🩸 Аналізи крові',
    catImaging:        '🫁 Візуалізація та ЕКГ',
    catFunctional:     '🧪 Функціональні та інші',
    catOther:          '🔬 Інше',
  },
  en: {
    // chat
    sessionStarted:    'Session started. The patient is waiting. Ask your first question.',
    finalDiag:         (d) => `Your final diagnosis: "${d}". Generating report…`,
    doctor:            'Doctor',
    patient:           'Patient',
    lab:               'Lab',
    system:            'System',
    step:              (n) => `Step ${n}`,
    question:          'Question',
    test:              'Test',
    revealed:          (name, pct) => `✅ Revealed: <strong>${name}</strong> <span style="opacity:.6;font-size:10px">(${pct}%)</span>`,
    noHints:           'No recommendations available',
    male:              'Male',
    female:            'Female',
    sex_m:             '♂ Patient',
    sex_f:             '♀ Patient',
    symptoms:          'symptoms',
    loading:           'Loading…',
    // splash
    splashLangLabel:   '🌐 Language for patient communication',
    splashInitStatus:  'Select a mode and click "Start"',
    splashInitializing:'Initializing system…',
    btnStart:          'Start session',
    modeSmartDesc:     'LLM-generated realistic responses (requires Ollama/Gemma)',
    modeSimpleDesc:    'Vector similarity without LLM. Fast and stable.',
    // topbar
    tbStep:            'Step',
    // left panel
    panelPatient:      'Patient',
    panelSymptoms:     'Detected symptoms',
    complaintsLabel:   '📋 Chief complaint',
    entropyLabel:      'Uncertainty (Entropy)',
    noSymptoms:        'Nothing detected yet',
    uncertaintyDown:   'uncertainty ↓',
    uncertaintyUp:     'uncertainty ↑',
    // input area
    inputPlaceholder:  'Ask the patient a question… (Enter to send)',
    btnAsk:            '➤ Ask',
    btnTest:           '🔬 Test <span id="test-counter" class="test-counter">0/3</span>',
    chipChestPain:     'Chest pain',
    chipBreath:        'Dyspnea',
    chipFever:         'Fever',
    chipFatigue:       'Fatigue',
    chipMeds:          'Medications',
    chipDiagnose:      '🩺 Make diagnosis',
    // hints panel
    hintsTitle:        '💡 Hints',
    hintsActive:       'Active',
    hintsLockTitle:    'Hints unavailable',
    hintsLockSub:      'Make',
    hintsLockSub2:     'more steps on your own',
    hintsIntro:        'The system recommends these questions based on the current diagnostic state:',
    hintsRefresh:      '↻ Refresh hints',
    hintsAnalyzing:    'Analyzing state…',
    // diagnosis dialog
    diagTitle:         '🩺 Final diagnosis',
    diagSub:           'Enter your diagnosis to finish the session and receive an evaluation',
    diagPlaceholder:   'E.g.: Spontaneous pneumothorax',
    btnCancel:         'Cancel',
    btnConfirmDiag:    'Confirm diagnosis',
    // report modal
    reportTitle:       'Competency Report',
    reportSubtitle:    'Diagnostic session results',
    btnNewSession:     '🔄 New session',
    resultCorrect:     'CORRECT DIAGNOSIS',
    resultWrong:       'INCORRECT DIAGNOSIS',
    resultYours:       'Yours:',
    resultTrue:        'Correct:',
    scoreTotal:        'Total score',
    scoreComm:         'Communication',
    scoreIG:           'Total IG',
    scoreCritical:     'Critical errors',
    tableAction:       'Action',
    tableScore:        'Score',
    // test dialog
    testDialogTitle:   '🔬 Order a test / examination',
    testDialogSub:     (left, max) => `Tests remaining: <strong>${left}</strong> of ${max}`,
    testSearchPH:      'Search test…',
    testSelected:      'Selected:',
    btnConfirmTest:    'Run test',
    testNoResults:     'Nothing found',
    testSearchResults: (n) => `🔍 Search results (${n})`,
    testLimitAlert:    'Test limit reached',
    testWaitAlert:     'Please wait for session initialization to complete.',
    // proc categories
    catBlood:          '🩸 Blood tests',
    catImaging:        '🫁 Imaging & ECG',
    catFunctional:     '🧪 Functional & other',
    catOther:          '🔬 Other',
  },
};
const t = () => L[selectedLang];

// ── Apply UI language to all static elements ──────────────
function applyUILang() {
  const l = t();
  // Splash
  safeSet('lang-label-text',    l.splashLangLabel);
  safeSet('load-status-init',   l.splashInitStatus);
  safeInnerHTML('btn-start',    l.btnStart);
  safeSet('mode-smart-desc',    l.modeSmartDesc);
  safeSet('mode-simple-desc',   l.modeSimpleDesc);
  // Topbar
  safeSet('tb-step-label',      l.tbStep);
  // Left panel
  safeSet('panel-patient-hdr',  l.panelPatient);
  safeSet('panel-symptoms-hdr', l.panelSymptoms);
  safeSet('complaint-label-el', l.complaintsLabel);
  safeSet('entropy-name-el',    l.entropyLabel);
  safeSet('no-symptoms-ph',     l.noSymptoms);
  // Input area
  const inp = document.getElementById('user-input');
  if (inp) inp.placeholder = l.inputPlaceholder;
  safeInnerHTML('btn-ask',      l.btnAsk);
  safeInnerHTML('btn-test',     l.btnTest);
  safeSet('chip-chest',         l.chipChestPain);
  safeSet('chip-breath',        l.chipBreath);
  safeSet('chip-fever',         l.chipFever);
  safeSet('chip-fatigue',       l.chipFatigue);
  safeSet('chip-meds',          l.chipMeds);
  safeSet('chip-diagnose',      l.chipDiagnose);
  // Hints panel
  safeSet('hints-panel-title',  l.hintsTitle);
  safeSet('hints-lock-title-el',l.hintsLockTitle);
  safeSet('hints-lock-sub-1',   l.hintsLockSub);
  safeSet('hints-lock-sub-2',   l.hintsLockSub2);
  safeSet('hints-intro-el',     l.hintsIntro);
  safeSet('hints-refresh-btn',  l.hintsRefresh);
  safeSet('hints-analyzing-el', l.hintsAnalyzing);
  // Diagnosis dialog
  safeSet('diag-dialog-title',  l.diagTitle);
  safeSet('diag-dialog-sub',    l.diagSub);
  const diagInp = document.getElementById('diag-input');
  if (diagInp) diagInp.placeholder = l.diagPlaceholder;
  safeSet('btn-cancel-diag',    l.btnCancel);
  safeSet('btn-confirm-diag',   l.btnConfirmDiag);
  // Report modal
  safeSet('report-modal-title', l.reportTitle);
  safeSet('report-modal-sub',   l.reportSubtitle);
  safeSet('btn-new-session',    l.btnNewSession);
  // Test dialog
  safeSet('test-dialog-title',  l.testDialogTitle);
  safeSet('test-search-input-ph', null);
  const testInp = document.getElementById('test-search-input');
  if (testInp) testInp.placeholder = l.testSearchPH;
  safeSet('test-selected-label-el', l.testSelected);
  safeSet('btn-cancel-test',    l.btnCancel);
  safeSet('btn-confirm-test-label', l.btnConfirmTest);
  updateTestDialogSub();
}

function updateTestDialogSub() {
  const el = document.getElementById('test-dialog-sub');
  if (el) el.innerHTML = t().testDialogSub(MAX_TESTS - testsUsed, MAX_TESTS);
}

function safeInnerHTML(id, html) {
  const el = document.getElementById(id);
  if (el) el.innerHTML = html;
}

// ── Hints Config ──────────────────────────────────────────
const HINTS_UNLOCK_AFTER = 3;
let hintsUnlocked = false;
let hintsRequested = false;

// ── Mode Selection ────────────────────────────────────────
function selectMode(mode) {
  selectedMode = mode;
  document.getElementById('card-smart').classList.toggle('active', mode === 'smart');
  document.getElementById('card-simple').classList.toggle('active', mode === 'simple');
}

// ── Language Selection ────────────────────────────────────
function selectLang(lang) {
  selectedLang = lang;
  document.getElementById('btn-lang-uk').classList.toggle('active', lang === 'uk');
  document.getElementById('btn-lang-en').classList.toggle('active', lang === 'en');
  applyUILang();
}

// Apply UI lang on page load (default: en)
document.addEventListener('DOMContentLoaded', applyUILang);

// ── Start Session ─────────────────────────────────────────
function startSession() {
  document.getElementById('btn-start').disabled = true;
  document.getElementById('dots').style.display = 'flex';
  safeSet('load-status', t().splashInitializing);
  socket.emit('init_session', { mode: selectedMode, lang: selectedLang });
}

socket.on('load_progress', data => {
  document.getElementById('load-status').textContent = data.msg;
});

socket.on('session_ready', data => {
  const lbl = t();
  applyUILang();
  // Populate patient info
  const isMale = data.sex === 'M';
  document.getElementById('pt-name').textContent =
    `${isMale ? lbl.sex_m : lbl.sex_f}, ${data.age} ${selectedLang === 'uk' ? 'р.' : 'y.o.'}`;
  document.getElementById('pt-meta').textContent =
    `${isMale ? lbl.male : lbl.female} · ${data.total_symptoms} ${lbl.symptoms}`;
  document.getElementById('pt-avatar').textContent = isMale ? '👨' : '👩';
  document.getElementById('pt-complaint').textContent = data.complaint;
  document.getElementById('tb-mode').textContent =
    data.mode === 'smart' ? 'Smart Patient' : 'Simple Patient';

  // Initial entropy
  updateEntropy(data.h0, null);

  diagnosisNames = data.diagnoses || [];
  const rawProcs  = data.procedures_raw  || data.procedures || [];
  const dispProcs = data.procedures || [];
  realProcedures = rawProcs.map((raw, i) => ({
    raw,
    display: dispProcs[i] || raw,
  }));
  TEST_CATALOG = buildCatalogFromProcedures(realProcedures);

  // Transition to app
  document.getElementById('splash').classList.add('fade-out');
  setTimeout(() => {
    document.getElementById('splash').style.display = 'none';
    document.getElementById('app').classList.add('visible');
  }, 500);

  addSystemMessage(lbl.sessionStarted);
});

// ── Input Handling ────────────────────────────────────────
function handleKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendQuestion();
  }
}

function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 120) + 'px';
}

function fillInput(text) {
  const inp = document.getElementById('user-input');
  inp.value = text;
  inp.focus();
  autoResize(inp);
}

function sendQuestion() {
  const inp = document.getElementById('user-input');
  const text = inp.value.trim();
  if (!text) return;
  inp.value = ''; autoResize(inp);
  addDoctorMessage(text, 'question');
  showTyping();
  socket.emit('ask_question', { text, type: 'question' });
}

// ── Test Dialog ───────────────────────────────────────────
const MAX_TESTS = 3;
let testsUsed = 0;
let selectedTest = null;
let filteredTestIndex = -1;
let realProcedures = [];
let TEST_CATALOG = [];
const PROC_CATEGORIES = [
  {
    key: 'catBlood',
    keywords: [
      'blood', 'glucose', 'crp', 'c-reactive', 'troponin', 'd-dimer',
      'liver', 'kidney', 'thyroid', 'coagulation', 'hemoglobin',
      'hematocrit', 'platelet', 'leukocyte', 'erythrocyte', 'ferritin',
      'cholesterol', 'triglyceride', 'albumin', 'bilirubin', 'creatinine',
      'urea', 'uric acid', 'culture', 'count', 'panel', 'serology',
      'antibod', 'antigen', 'marker', 'lactate', 'procalcitonin',
      'complete blood', 'cbc ', 'wbc', 'rbc', 'esr', 'psa',
      'hba1c', 'calcium', 'sodium', 'potassium', 'chloride', 'phosphate',
      'magnesium', 'iron', 'transferrin', 'ldh', 'ck ', 'ast', 'alt',
      'ggt', 'bnp', 'nt-pro', 'amylase', 'lipase', 'cortisol',
      'insulin', 'vitamin', 'folate', 'b12', 'zinc', 'copper',
      'immunoglobulin', 'complement', 'ana ', 'anca', 'rf ', 'rheumatoid',
      'strep', 'monospot', 'hiv', 'hepatitis', 'syphilis', 'toxo',
      'culture', 'sensitivity', 'pcr', 'rapid test', 'ag test',
    ],
    icon: '🩸',
  },
  {
    key: 'catImaging',
    keywords: [
      'x-ray', 'xray', 'radiograph', 'ct ', 'ct-', 'scan', 'mri',
      'ultrasound', 'echo', 'echocardiograph', 'angiograph',
      'scintigraph', 'pet ', 'doppler', 'fluoroscop', 'mammograph',
      'endoscop', 'bronchoscop', 'colonoscop', 'ecg', 'ekg',
      'electrocardiograph', 'holter', 'imaging', 'chest film',
      'chest radiograph', 'abdominal film', 'bone scan', 'dexa',
      'lung scan', 'ventilation', 'perfusion', 'mra ', 'mrcp',
      'ercp', 'cystoscop', 'arthroscop', 'laparoscop', 'thoracoscop',
    ],
    icon: '🫁',
  },
  {
    key: 'catFunctional',
    keywords: [
      'spirometr', 'peak flow', 'spo2', 'oxygen saturat', 'oximetr',
      'temperature', 'pulse', 'pressure', 'urine', 'urinalysis',
      'stool', 'feces', 'swab', 'smear', 'biopsy', 'sputum',
      'pleural', 'lumbar', 'puncture', 'skin test', 'allergy test',
      'measure', 'level', 'rate', 'audiometr', 'visual', 'reflex',
      'tilt', 'stress test', 'pulmonary function', 'flow rate',
      'fev1', 'fvc', 'forced', 'exhaled', 'breath test',
      'sweat test', 'schilling', 'colonoscopy prep', 'enema',
      'ph monitor', 'manometr', 'tympanometr', 'audiogram',
    ],
    icon: '🧪',
  },
];

const SYMPTOM_BLOOD_HINTS = [
  'blood in', 'blood clot', 'red blood', 'dark stool', 'black stool',
  'bleeding', 'hemorrh', 'blood pressure', 'hypertension',
];
const SYMPTOM_IMAGING_HINTS = [
  'contact with', 'exposure to', 'been in contact', 'two images', 'double vision',
  'pertussis', 'whooping', 'allerg', 'hiv-infected', 'immunodeficien',
];
const SYMPTOM_FUNCTIONAL_HINTS = [
  'fever', 'temperature', 'measured', 'thermometer', 'diarrhea', 'stool frequency',
  'pass stool', 'pass gas', 'bowel', 'vomit', 'nausea', 'sputum', 'cough up',
  'breathe', 'breathing', 'asthma', 'bronchodilat', 'copd', 'oxygen',
  'saturat', 'pulse', 'palpitat', 'sweat', 'weight', 'appetite',
];

function categorizeProcedureRaw(name) {
  const low = name.toLowerCase();

  for (const cat of PROC_CATEGORIES) {
    if (cat.keywords.some(kw => low.includes(kw))) {
      return { catKey: cat.key, icon: cat.icon };
    }
  }

  if (SYMPTOM_BLOOD_HINTS.some(kw => low.includes(kw))) {
    return { catKey: 'catBlood', icon: '🩸' };
  }
  if (SYMPTOM_IMAGING_HINTS.some(kw => low.includes(kw))) {
    return { catKey: 'catImaging', icon: '🫁' };
  }
  if (SYMPTOM_FUNCTIONAL_HINTS.some(kw => low.includes(kw))) {
    return { catKey: 'catFunctional', icon: '🧪' };
  }

  return { catKey: 'catOther', icon: '🔬' };
}

function categorizeProcedure(name) {
  const low = name.toLowerCase();
  for (const cat of PROC_CATEGORIES) {
    if (cat.keywords.some(kw => low.includes(kw))) {
      return { cat: t()[cat.key], icon: cat.icon };
    }
  }
  return { cat: t().catOther, icon: '🔬' };
}

function buildCatalogFromProcedures(procs) {
  const groups = {};
  const CAT_KEYS = ['catBlood', 'catImaging', 'catFunctional', 'catOther'];
  for (const proc of procs) {
    const displayName = (typeof proc === 'object') ? proc.display : proc;
    const rawName     = (typeof proc === 'object') ? proc.raw     : proc;
    const { catKey, icon } = categorizeProcedureRaw(rawName);
    if (!groups[catKey]) groups[catKey] = { items: [] };
    groups[catKey].items.push({ icon, name: displayName, raw: rawName });
  }
  return CAT_KEYS
    .filter(k => groups[k])
    .map(k => ({ cat: t()[k], items: groups[k].items }))
    .concat(
      Object.entries(groups)
        .filter(([k]) => !CAT_KEYS.includes(k))
        .map(([, g]) => ({ cat: t().catOther, items: g.items }))
    );
}

function openTestDialog() {
  if (testsUsed >= MAX_TESTS) return;
  if (!realProcedures.length) {
    alert(t().testWaitAlert);
    return;
  }
  selectedTest = null;
  filteredTestIndex = -1;
  document.getElementById('test-search-input').value = '';
  updateTestSelectedRow();
  updateTestDialogSub();
  TEST_CATALOG = buildCatalogFromProcedures(realProcedures);
  renderTestCategories(TEST_CATALOG);
  document.getElementById('test-dialog').classList.add('open');
  setTimeout(() => document.getElementById('test-search-input').focus(), 120);
}

function closeTestDialog() {
  document.getElementById('test-dialog').classList.remove('open');
  selectedTest = null;
}

function renderTestCategories(catalog) {
  const container = document.getElementById('test-categories');
  if (!catalog.length) {
    container.innerHTML = `<div style="color:var(--text-muted);font-size:13px;text-align:center;padding:16px 0">${t().testNoResults}</div>`;
    return;
  }
  container.innerHTML = catalog.map(group => `
    <div>
      <div class="test-category-title">${escHtml(group.cat)}</div>
      <div class="test-chips-row">
        ${group.items.map(item => {
          const rawName  = item.raw || item.name;
          const dispName = item.name;
          const isSelected = selectedTest && selectedTest.raw === rawName;
          return `<div class="test-chip${isSelected ? ' selected' : ''}"
            onclick="selectTest('${rawName.replace(/'/g,"\\'")}','${dispName.replace(/'/g,"\\'")}')">
            <span class="chip-icon">${item.icon}</span>${escHtml(dispName)}
          </div>`;
        }).join('')}
      </div>
    </div>`).join('');
}

function filterTests(q) {
  filteredTestIndex = -1;
  if (!q.trim()) {
    renderTestCategories(TEST_CATALOG);
    return;
  }
  const low = q.toLowerCase();
  const matched = realProcedures.filter(p => {
    const display = (typeof p === 'object') ? p.display : p;
    const raw     = (typeof p === 'object') ? p.raw     : p;
    return display.toLowerCase().includes(low) || raw.toLowerCase().includes(low);
  });
  if (!matched.length) {
    document.getElementById('test-categories').innerHTML =
      `<div style="color:var(--text-muted);font-size:13px;text-align:center;padding:16px 0">${t().testNoResults}</div>`;
    return;
  }
  const items = matched.slice(0, 20).map(p => {
    const displayName = (typeof p === 'object') ? p.display : p;
    const rawName     = (typeof p === 'object') ? p.raw     : p;
    const { icon } = categorizeProcedureRaw(rawName);
    return { icon, name: displayName, raw: rawName };
  });
  renderTestCategories([{ cat: t().testSearchResults(matched.length), items }]);
}

function testInputKey(e) {
  if (e.key === 'Escape') { closeTestDialog(); return; }
  if (e.key === 'Enter' && selectedTest) { confirmTest(); return; }
}

function selectTest(rawName, displayName) {
  selectedTest = { raw: rawName, display: displayName || rawName };
  updateTestSelectedRow();
  const q = document.getElementById('test-search-input').value;
  filterTests(q);
}

function clearSelectedTest() {
  selectedTest = null;
  updateTestSelectedRow();
  const q = document.getElementById('test-search-input').value;
  filterTests(q);
}

function updateTestSelectedRow() {
  const row = document.getElementById('test-selected-row');
  const nameEl = document.getElementById('test-selected-name');
  const confirmBtn = document.getElementById('btn-confirm-test');
  if (selectedTest) {
    row.style.display = 'flex';
    nameEl.textContent = selectedTest.display || selectedTest.raw || selectedTest;
    confirmBtn.disabled = false;
  } else {
    row.style.display = 'none';
    confirmBtn.disabled = true;
  }
}

function confirmTest() {
  if (!selectedTest || testsUsed >= MAX_TESTS) return;
  const rawText     = selectedTest.raw     || selectedTest;
  const displayText = selectedTest.display || selectedTest;
  closeTestDialog();
  testsUsed++;
  updateTestCounter();
  const inp = document.getElementById('user-input');
  inp.value = '';
  addDoctorMessage(displayText, 'test');
  showTyping();
  socket.emit('ask_question', { text: rawText, type: 'test' });
}

function updateTestCounter() {
  const el = document.getElementById('test-counter');
  const btn = document.getElementById('btn-test');
  if (el) {
    el.textContent = `${testsUsed}/${MAX_TESTS}`;
    if (testsUsed >= MAX_TESTS) {
      el.classList.add('exhausted');
      btn.disabled = true;
      btn.title = t().testLimitAlert;
    }
  }
  updateTestDialogSub();
}

function sendAsTest() {
  openTestDialog();
}

// ── Messages ──────────────────────────────────────────────
function addDoctorMessage(text, type) {
  const lbl = t();
  clientStep += 1;
  const badge = type === 'test'
    ? `<span class="msg-type-badge badge-test">${lbl.test}</span>`
    : `<span class="msg-type-badge badge-question">${lbl.question}</span>`;
  appendMsg('doctor', '🩺', lbl.doctor, badge, text, lbl.step(clientStep));
}

function addPatientMessage(text, newSymptoms) {
  const html = buildPatientHTML(text, newSymptoms);
  appendRawMsg('patient', '😷', t().patient, '', html);
}

function addLabMessage(text) {
  appendMsg('lab', '🔬', t().lab, '', text, '');
}

function addSystemMessage(text) {
  appendMsg('system', 'ℹ️', t().system, '', text, '');
}

function buildPatientHTML(text, newSymptoms) {
  let html = `<div class="msg-text">${escHtml(text)}</div>`;
  if (newSymptoms && newSymptoms.length) {
    newSymptoms.forEach(([name, sim]) => {
      html += `<div class="symptom-reveal">${t().revealed(escHtml(name), (sim*100).toFixed(0))}</div>`;
    });
  }
  return html;
}

function appendMsg(cls, icon, sender, badge, text, stepLabel) {
  const container = document.getElementById('chat-messages');
  const div = document.createElement('div');
  div.className = `msg ${cls}`;
  div.innerHTML = `
    <div class="msg-avatar">${icon}</div>
    <div class="msg-body">
      <div class="msg-header">
        <span class="msg-sender">${sender}</span>
        ${badge}
        <span class="msg-step">${stepLabel}</span>
      </div>
      <div class="msg-text">${escHtml(text)}</div>
    </div>`;
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
}

function appendRawMsg(cls, icon, sender, badge, innerHtml) {
  const container = document.getElementById('chat-messages');
  const div = document.createElement('div');
  div.className = `msg ${cls}`;
  div.innerHTML = `
    <div class="msg-avatar">${icon}</div>
    <div class="msg-body">
      <div class="msg-header">
        <span class="msg-sender">${sender}</span>${badge}
      </div>
      ${innerHtml}
    </div>`;
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
}

let typingEl = null;
function showTyping() {
  removeTyping();
  const container = document.getElementById('chat-messages');
  typingEl = document.createElement('div');
  typingEl.className = 'msg patient';
  typingEl.id = 'typing';
  typingEl.innerHTML = `<div class="msg-avatar">😷</div>
    <div class="msg-body"><div class="typing-indicator"><span></span><span></span><span></span></div></div>`;
  container.appendChild(typingEl);
  container.scrollTop = container.scrollHeight;
}
function removeTyping() {
  if (typingEl) { typingEl.remove(); typingEl = null; }
}

function translateLabResult(text) {
  if (!text || selectedLang !== 'uk') return text;

  let m = text.match(/^Results for '(.+?)': Abnormal findings consistent with (.+)\.$/);
  if (m) {
    return `Результати '${m[1]}': Патологічні зміни, характерні для ${m[2]}.`;
  }

  m = text.match(/^Results for '(.+?)': Findings are within normal range\.$/);
  if (m) {
    return `Результати '${m[1]}': Показники в межах норми.`;
  }

  m = text.match(/^I'm not sure what '(.+?)' refers to\.$/);
  if (m) {
    return `Не зрозуміло, що означає '${m[1]}'.`;
  }

  if (text.includes('Please specify which test')) {
    return 'Будь ласка, вкажіть, який аналіз ви хочете провести.';
  }

  return text;
}

socket.on('answer', data => {
  removeTyping();
  if (data.type === 'test') {
    let labText = data.answer;
    if (selectedLang === 'uk' && labText) {
      labText = translateLabResult(labText);
    }
    addLabMessage(labText);
  } else {
    addPatientMessage(data.answer, data.newly_revealed);
  }

  // Update entropy
  updateEntropy(data.h_new, data.ig);

  // Update top diagnoses
  if (data.top_diagnoses) updateDiagnoses(data.top_diagnoses);

  // Sync clientStep with server to avoid drift
  clientStep = data.step || clientStep;

  // Update symptoms list
  if (data.newly_revealed && data.newly_revealed.length) {
    data.newly_revealed.forEach(([name, sim]) => addSymptom(name, sim));
    sessionStats.symptoms += data.newly_revealed.length;
    safeSet('st-symptoms', sessionStats.symptoms);
  }

  // Update step
  const step = data.step || 1;
  document.getElementById('tb-step').textContent = step;
  updateHintsProgress(step);

  // Stats
  if (data.type === 'test') sessionStats.tests++;
  else sessionStats.questions++;
  sessionStats.ig += (data.ig || 0);
  safeSet('st-questions', sessionStats.questions);
  safeSet('st-tests', sessionStats.tests);
  safeSet('st-ig', sessionStats.ig.toFixed(4));

  // Re-enable input
  document.getElementById('btn-ask').disabled = false;
  document.getElementById('user-input').disabled = false;
});

// ── Entropy ───────────────────────────────────────────────
function updateEntropy(h, ig) {
  const H_MAX = 8.0;
  const pct = Math.min(h / H_MAX * 100, 100);
  document.getElementById('ent-val').textContent = h.toFixed(3);
  document.getElementById('ent-bar').style.width = pct + '%';
  document.getElementById('tb-entropy').textContent = h.toFixed(3);

  if (ig !== null && ig !== undefined) {
    const row = document.getElementById('delta-h-row');
    row.style.display = 'flex';
    document.getElementById('dh-icon').textContent = ig >= 0 ? '↓' : '↑';
    document.getElementById('dh-icon').style.color = ig >= 0 ? 'var(--green)' : 'var(--red)';
    document.getElementById('dh-val').textContent = Math.abs(ig).toFixed(4);
    document.getElementById('dh-val').style.color = ig >= 0 ? 'var(--green)' : 'var(--red)';
    document.getElementById('dh-label').textContent =
      ig >= 0 ? t().uncertaintyDown : t().uncertaintyUp;
  }
}

// ── Diagnoses ─────────────────────────────────────────────
function updateDiagnoses(topDiags) {
  const el = document.getElementById('diag-list');
  if (!el) return;
  el.innerHTML = '';
  topDiags.forEach((d, i) => {
    const rankClass = i === 0 ? 'rank-1' : i === 1 ? 'rank-2' : i === 2 ? 'rank-3' : 'rank-other';
    const div = document.createElement('div');
    div.className = `diag-item ${rankClass}`;
    div.innerHTML = `
      <div class="diag-header">
        <span class="diag-rank">${i+1}</span>
        <span class="diag-name">${escHtml(d.name)}</span>
        <span class="diag-pct">${(d.prob * 100).toFixed(1)}%</span>
      </div>
      <div class="diag-bar-track">
        <div class="diag-bar-fill" style="width:${d.prob * 100}%"></div>
      </div>`;
    el.appendChild(div);
  });
}

// ── Symptoms ──────────────────────────────────────────────
function addSymptom(name, sim) {
  const el = document.getElementById('symptoms-list');
  const ph = el.querySelector('[style*="color:var(--text-muted)"]');
  if (ph) ph.remove();

  const div = document.createElement('div');
  div.className = 'symptom-item';
  div.innerHTML = `<div class="symptom-dot"></div>
    <span class="symptom-name">${escHtml(name)}</span>
    <span class="symptom-sim">${(sim*100).toFixed(0)}%</span>`;
  el.appendChild(div);
}

// ── Diagnosis Dialog ──────────────────────────────────────
let diagSugIndex = -1;

function openDiagDialog() {
  document.getElementById('diag-dialog').classList.add('open');
  diagSugIndex = -1;
  setTimeout(() => document.getElementById('diag-input').focus(), 100);
}
function closeDiagDialog() {
  document.getElementById('diag-dialog').classList.remove('open');
  closeDiagSuggestions();
}
function confirmDiagnosis() {
  const diag = document.getElementById('diag-input').value.trim();
  if (!diag) return;
  closeDiagDialog();
  socket.emit('finalize', { diagnosis: diag });
  addSystemMessage(t().finalDiag(diag));
}

function diagInputChange(val) {
  diagSugIndex = -1;
  const q = val.trim().toLowerCase();
  const box = document.getElementById('diag-suggestions');
  if (!q || diagnosisNames.length === 0) {
    closeDiagSuggestions();
    return;
  }
  const matches = diagnosisNames
    .filter(d => d.toLowerCase().includes(q))
    .slice(0, 8);
  if (!matches.length) {
    closeDiagSuggestions();
    return;
  }
  box.innerHTML = matches.map((d, i) => {
    const idx = d.toLowerCase().indexOf(q);
    const before = escHtml(d.slice(0, idx));
    const match  = escHtml(d.slice(idx, idx + q.length));
    const after  = escHtml(d.slice(idx + q.length));
    return `<div class="diag-suggestion-item" data-idx="${i}"
      onmousedown="pickDiagSuggestion('${d.replace(/'/g,"\\'")}')">
      <span class="sug-icon">🩺</span>
      <span>${before}<span class="sug-match">${match}</span>${after}</span>
    </div>`;
  }).join('');
  box.classList.add('open');
}

function diagInputKey(e) {
  const box = document.getElementById('diag-suggestions');
  const items = box.querySelectorAll('.diag-suggestion-item');
  if (box.classList.contains('open') && items.length) {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      diagSugIndex = Math.min(diagSugIndex + 1, items.length - 1);
      highlightSugItem(items);
      return;
    }
    if (e.key === 'ArrowUp') {
      e.preventDefault();
      diagSugIndex = Math.max(diagSugIndex - 1, -1);
      highlightSugItem(items);
      return;
    }
    if (e.key === 'Enter') {
      if (diagSugIndex >= 0 && items[diagSugIndex]) {
        e.preventDefault();
        items[diagSugIndex].dispatchEvent(new MouseEvent('mousedown'));
        return;
      }
    }
    if (e.key === 'Escape') {
      closeDiagSuggestions();
      return;
    }
  }
  if (e.key === 'Enter') confirmDiagnosis();
}

function highlightSugItem(items) {
  items.forEach((el, i) => el.classList.toggle('active', i === diagSugIndex));
  if (diagSugIndex >= 0) items[diagSugIndex].scrollIntoView({ block: 'nearest' });
}

function pickDiagSuggestion(name) {
  document.getElementById('diag-input').value = name;
  closeDiagSuggestions();
  document.getElementById('diag-input').focus();
}

function closeDiagSuggestions() {
  const box = document.getElementById('diag-suggestions');
  if (box) { box.classList.remove('open'); box.innerHTML = ''; }
}

document.addEventListener('mousedown', e => {
  const wrap = document.querySelector('.diag-input-wrap');
  if (wrap && !wrap.contains(e.target)) closeDiagSuggestions();
});

socket.on('report', data => {
  showReport(data);
});

// ── Report ────────────────────────────────────────────────
function showReport(data) {
  const l = t();
  const isCorrect = data.correct;
  const resultClass = isCorrect ? 'result-correct' : 'result-wrong';
  const resultIcon = isCorrect ? '✅' : '❌';
  const resultText = isCorrect ? l.resultCorrect : l.resultWrong;

  let stepsHTML = '';
  (data.steps || []).forEach(s => {
    const scoreClass = s.total_step >= 0 ? 'score-pos' : 'score-neg';
    const penClass = s.penalty > 0 ? 'score-neg' : '';
    stepsHTML += `<tr>
      <td>${s.step}</td>
      <td>${escHtml(s.query.length > 45 ? s.query.slice(0,45)+'…' : s.query)}</td>
      <td>${s.ig !== undefined ? (s.ig >= 0 ? '+' : '') + s.ig.toFixed(3) : '—'}</td>
      <td class="${penClass}">${s.penalty > 0 ? '⚠ ' + s.penalty.toFixed(2) : '—'}</td>
      <td class="${scoreClass}">${s.total_step.toFixed(2)}</td>
    </tr>`;
  });

  const html = `
    <div class="result-banner ${resultClass}">
      <div class="result-icon">${resultIcon}</div>
      <div>
        <div class="result-main">${resultText}</div>
        <div class="result-sub">
          ${l.resultYours} <strong>${escHtml(data.user_diagnosis)}</strong>
          &nbsp;·&nbsp; ${l.resultTrue} <strong>${escHtml(data.true_diagnosis)}</strong>
        </div>
      </div>
    </div>
    <div class="summary-cards">
      <div class="summary-card">
        <div class="summary-num">${data.total_score?.toFixed(2) ?? '—'}</div>
        <div class="summary-lbl">${l.scoreTotal}</div>
      </div>
      <div class="summary-card">
        <div class="summary-num">${data.avg_comm?.toFixed(2) ?? '—'}</div>
        <div class="summary-lbl">${l.scoreComm}</div>
      </div>
      <div class="summary-card">
        <div class="summary-num">${data.total_ig?.toFixed(3) ?? '—'}</div>
        <div class="summary-lbl">${l.scoreIG}</div>
      </div>
      <div class="summary-card">
        <div class="summary-num">${data.critical_errors ?? 0}</div>
        <div class="summary-lbl">${l.scoreCritical}</div>
      </div>
    </div>
    <table class="report-table">
      <thead><tr>
        <th>#</th><th>${l.tableAction}</th><th>ΔH</th><th>Penalty</th><th>${l.tableScore}</th>
      </tr></thead>
      <tbody>${stepsHTML}</tbody>
    </table>`;

  document.getElementById('report-content').innerHTML = html;
  document.getElementById('report-modal').classList.add('open');
}

// ── Hints System ─────────────────────────────────────────
function updateHintsProgress(step) {
  if (hintsUnlocked) return;

  const pct = Math.min((step / HINTS_UNLOCK_AFTER) * 100, 100);
  const left = Math.max(HINTS_UNLOCK_AFTER - step, 0);

  document.getElementById('hints-progress-fill').style.width = pct + '%';
  document.getElementById('hints-steps-left').textContent = left;

  if (step >= HINTS_UNLOCK_AFTER && !hintsUnlocked) {
    hintsUnlocked = true;
    unlockHints();
  }
}

function unlockHints() {
  const lockEl = document.getElementById('hints-lock');
  const contentEl = document.getElementById('hints-content');
  const badge = document.getElementById('hints-badge');

  lockEl.classList.add('hints-unlock-anim');
  setTimeout(() => {
    lockEl.style.display = 'none';
    contentEl.style.display = 'block';
    badge.style.display = 'inline-block';
    badge.textContent = t().hintsActive;
    requestHints();
  }, 600);
}

function requestHints() {
  if (!hintsUnlocked) return;
  document.getElementById('hints-loading').style.display = 'flex';
  document.getElementById('hints-list').style.opacity = '0.3';
  socket.emit('get_hints', { n: 3 });
}

socket.on('hints_ready', data => {
  document.getElementById('hints-loading').style.display = 'none';
  const listEl = document.getElementById('hints-list');
  listEl.style.opacity = '1';
  listEl.innerHTML = '';

  if (!data.hints || data.hints.length === 0) {
    listEl.innerHTML = `<div style="color:var(--text-muted);font-size:12px">${t().noHints}</div>`;
    return;
  }

  data.hints.forEach((hint, i) => {
    const isTest = hint.type === 'test';
    const igColor = hint.ig >= 0 ? 'var(--green)' : 'var(--red)';
    const igSign = hint.ig >= 0 ? '−' : '+';
    const igAbs = Math.abs(hint.ig).toFixed(4);

    const div = document.createElement('div');
    div.className = 'hint-item';
    div.style.animationDelay = (i * 0.08) + 's';
    div.innerHTML = `
      <div class="hint-rank">${i + 1}</div>
      <div class="hint-body">
        <div class="hint-text">${escHtml(hint.text)}</div>
        <div class="hint-meta">
          ${isTest ? `<span class="hint-type-badge badge-test">${t().test}</span>` : `<span class="hint-type-badge badge-question">${t().question}</span>`}
          <span class="hint-ig" style="color:${igColor}">ΔH ${igSign}${igAbs}</span>
        </div>
      </div>
      <button class="hint-use-btn" onclick="fillInput(\`${hint.text.replace(/`/g, "'")}\`)" title="Use this question">↗</button>
    `;
    listEl.appendChild(div);
  });
});

function escHtml(s) {
  return String(s)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function safeSet(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value;
}