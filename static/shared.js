// Tailwind shared config
if (typeof tailwind !== 'undefined') {
    tailwind.config = {
        darkMode: "class",
        theme: {
            extend: {
                colors: {
                    "outline-variant": "#adb3b4",
                    "primary-container": "#d8e2ff",
                    "on-tertiary-fixed-variant": "#634900",
                    "surface-variant": "#dde4e5",
                    "surface-dim": "#d4dbdd",
                    "on-surface": "#2d3435",
                    "inverse-on-surface": "#9c9d9d",
                    "on-background": "#2d3435",
                    "primary-fixed-dim": "#c2d4ff",
                    "on-secondary-fixed": "#264350",
                    "tertiary-dim": "#6b4e00",
                    "on-error-container": "#752121",
                    "on-tertiary-fixed": "#402d00",
                    primary: "#385da0",
                    "secondary-fixed": "#c9e7f7",
                    background: "#f9f9f9",
                    "tertiary-container": "#fec330",
                    "inverse-primary": "#93b6ff",
                    "surface-container-highest": "#dde4e5",
                    "error-container": "#fe8983",
                    "on-error": "#fff7f6",
                    "error-dim": "#4e0309",
                    tertiary: "#7a5a00",
                    "on-tertiary-container": "#584000",
                    "on-secondary": "#f3faff",
                    "surface-bright": "#f9f9f9",
                    surface: "#f9f9f9",
                    "primary-fixed": "#d8e2ff",
                    "secondary-fixed-dim": "#bbd9e9",
                    "surface-container": "#ebeeef",
                    error: "#9f403d",
                    "secondary-container": "#c9e7f7",
                    "surface-container-high": "#e4e9ea",
                    "inverse-surface": "#0c0f0f",
                    "on-primary-container": "#2a5092",
                    "on-secondary-fixed-variant": "#435f6d",
                    "on-primary-fixed": "#103d7e",
                    "tertiary-fixed": "#fec330",
                    "surface-tint": "#385da0",
                    "on-surface-variant": "#5a6061",
                    "primary-dim": "#2a5193",
                    secondary: "#466370",
                    "on-secondary-container": "#395663",
                    "surface-container-low": "#f2f4f4",
                    outline: "#757c7d",
                    "on-primary-fixed-variant": "#355a9d",
                    "on-tertiary": "#fff8f1",
                    "surface-container-lowest": "#ffffff",
                    "secondary-dim": "#3a5764",
                    "tertiary-fixed-dim": "#efb520",
                    "on-primary": "#f7f7ff",
                },
                fontFamily: {
                    headline: ["Manrope"],
                    body: ["Work Sans"],
                    label: ["Inter"],
                },
                borderRadius: {
                    DEFAULT: "0.125rem",
                    lg: "0.25rem",
                    xl: "0.5rem",
                    full: "0.75rem",
                },
            },
        },
    };
}

// Escape a string for safe insertion into innerHTML
function escapeHtml(text) {
    return String(text === null || text === undefined ? "" : text)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
}

// Escape text and colorize bridge suit symbols
function colorSuits(text) {
    return escapeHtml(text)
        .replace(/♣/g, '<span style="color:green;">♣</span>')
        .replace(/♦/g, '<span style="color:red;">♦</span>')
        .replace(/♥/g, '<span style="color:red;">♥</span>')
        .replace(/♠/g, '<span style="color:green;">♠</span>');
}

// Small transient notification at the bottom of the screen
function showToast(message, isError) {
    const existing = document.getElementById("app-toast");
    if (existing) existing.remove();
    const toast = document.createElement("div");
    toast.id = "app-toast";
    toast.className = "app-toast" + (isError ? " app-toast-error" : "");
    toast.textContent = message;
    document.body.appendChild(toast);
    setTimeout(() => toast.classList.add("show"), 10);
    setTimeout(() => {
        toast.classList.remove("show");
        setTimeout(() => toast.remove(), 300);
    }, 2200);
}

// ---- Edit protection (passcode) ----

window.appUnlocked = false;

// Query the server for the current auth state, update UI and notify pages
async function refreshAuthState() {
    try {
        const res = await fetch("/api/auth");
        const data = await res.json();
        window.appUnlocked = !!data.unlocked;
    } catch (e) {
        window.appUnlocked = false;
    }
    applyAuthUI();
    document.dispatchEvent(new CustomEvent("authchange"));
}

// Show/hide elements that require an unlocked session
function applyAuthUI() {
    const authArea = document.getElementById("authArea");
    if (authArea) {
        authArea.innerHTML = window.appUnlocked
            ? `<button onclick="lockNow()" title="Lock editing" class="flex items-center gap-1 px-3 py-1.5 bg-surface-container text-on-surface-variant rounded font-label text-sm hover:bg-surface-container-high transition-colors">
                    <span class="material-symbols-outlined text-sm">lock</span>
                    Lock
               </button>`
            : `<button onclick="showUnlockDialog()" title="Unlock editing" class="flex items-center gap-1 px-3 py-1.5 bg-tertiary-container/40 text-on-tertiary-container rounded font-label text-sm hover:bg-tertiary-container/70 transition-colors">
                    <span class="material-symbols-outlined text-sm">lock_open</span>
                    Unlock
               </button>`;
    }
    document.querySelectorAll("[data-auth-required]").forEach((el) => {
        el.style.display = window.appUnlocked ? "" : "none";
    });
}

function showUnlockDialog() {
    if (document.getElementById("unlockOverlay")) return;
    const overlay = document.createElement("div");
    overlay.id = "unlockOverlay";
    overlay.className = "unlock-overlay";
    overlay.innerHTML = `
        <div class="unlock-card">
            <span class="material-symbols-outlined text-primary text-3xl mb-2">lock</span>
            <h2 class="font-headline font-bold text-on-surface text-lg mb-1">Editing is protected</h2>
            <p class="text-sm text-on-surface-variant mb-4">Enter the passcode to create, edit or delete systems.</p>
            <form id="unlockForm">
                <input id="unlockPass" type="password" autocomplete="current-password" placeholder="Passcode" class="unlock-input" />
                <div id="unlockError" class="text-xs mt-2" style="color:#9f403d; display:none;">Incorrect passcode</div>
                <div class="flex justify-end gap-2 mt-4">
                    <button type="button" id="unlockCancel" class="px-4 py-2 bg-surface-container text-on-surface-variant rounded font-label text-sm hover:bg-surface-container-high transition-colors">Cancel</button>
                    <button type="submit" class="px-4 py-2 bg-primary text-on-primary rounded font-label text-sm hover:bg-primary-dim transition-colors">Unlock</button>
                </div>
            </form>
        </div>
    `;
    document.body.appendChild(overlay);
    const input = overlay.querySelector("#unlockPass");
    input.focus();
    overlay.querySelector("#unlockCancel").onclick = closeUnlockDialog;
    overlay.addEventListener("click", (e) => {
        if (e.target === overlay) closeUnlockDialog();
    });
    overlay.querySelector("#unlockForm").onsubmit = async (e) => {
        e.preventDefault();
        const res = await fetch("/api/login", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ passcode: input.value }),
        });
        if (res.ok) {
            closeUnlockDialog();
            showToast("Editing unlocked");
            refreshAuthState();
        } else {
            overlay.querySelector("#unlockError").style.display = "block";
            input.value = "";
            input.focus();
        }
    };
}

function closeUnlockDialog() {
    const overlay = document.getElementById("unlockOverlay");
    if (overlay) overlay.remove();
}

async function lockNow() {
    await fetch("/api/logout", { method: "POST" });
    showToast("Editing locked");
    refreshAuthState();
}

// Delete the system identified by the current URL and go back home
async function deleteCurrentTree() {
    const id = window.location.pathname.split("/").pop();
    if (!confirm("Delete this system? This cannot be undone.")) return;
    const res = await fetch("/api/tree/" + id, { method: "DELETE" });
    if (res.ok) {
        window.treeDeleted = true;
        window.location.href = "/";
    } else if (res.status === 401) {
        showToast("Passcode required", true);
        showUnlockDialog();
    } else {
        showToast("Delete failed", true);
    }
}

// Render the common page layout
function renderLayout(options) {
    // options: { mode: 'view' | 'edit' | 'practice' | 'index', title: '', titleId: '', subtitle: '', badge: '', badgeColor: '' }
    const mode = options.mode || 'index';
    const badgeColor = options.badgeColor || (mode === 'edit' ? 'bg-primary-container text-on-primary-container' : 'bg-tertiary-container text-on-tertiary-container');
    const modeLabel = mode === 'edit' ? 'Edit Mode' : mode === 'view' ? 'View Mode' : mode === 'practice' ? 'Practice Mode' : 'Home';

    const btnGhost = "flex items-center gap-2 px-4 py-1.5 bg-surface-container-highest text-on-surface rounded font-label text-sm hover:bg-surface-container-high transition-colors no-underline";
    const btnPrimary = "flex items-center gap-2 px-4 py-1.5 bg-primary text-on-primary rounded font-label text-sm hover:bg-primary-dim transition-colors no-underline";
    const btnDanger = "flex items-center gap-2 px-3 py-1.5 bg-error-container/30 text-on-error-container rounded font-label text-sm hover:bg-error-container/60 transition-colors";
    const deleteBtn = `
        <button onclick="deleteCurrentTree()" title="Delete system" data-auth-required style="display:none;" class="${btnDanger}">
            <span class="material-symbols-outlined text-sm">delete</span>
            Delete
        </button>
    `;

    let headerActions = '';
    if (mode === 'edit') {
        headerActions = `
            <button onclick="saveTree()" data-auth-required style="display:none;" class="${btnGhost}">
                <span class="material-symbols-outlined text-sm">save</span>
                Save
            </button>
            <a id="viewLink" href="#" class="${btnGhost}">
                <span class="material-symbols-outlined text-sm">visibility</span>
                View
            </a>
            <a id="practiceLink" href="#" class="${btnPrimary}">
                <span class="material-symbols-outlined text-sm">school</span>
                Practice
            </a>
            ${deleteBtn}
        `;
    } else if (mode === 'view') {
        headerActions = `
            <a id="practiceLink" href="#" class="${btnPrimary}">
                <span class="material-symbols-outlined text-sm">school</span>
                Practice
            </a>
            <a id="editLink" href="#" data-auth-required style="display:none;" class="${btnGhost}">
                <span class="material-symbols-outlined text-sm">edit</span>
                Edit
            </a>
            ${deleteBtn}
        `;
    } else if (mode === 'practice') {
        headerActions = `
            <a id="viewLink" href="#" class="${btnGhost}">
                <span class="material-symbols-outlined text-sm">visibility</span>
                View
            </a>
            <a id="editLink" href="#" data-auth-required style="display:none;" class="${btnGhost}">
                <span class="material-symbols-outlined text-sm">edit</span>
                Edit
            </a>
            ${deleteBtn}
        `;
    }

    return `
        <!-- TopAppBar -->
        <header class="fixed top-0 z-50 bg-[#ffffff] dark:bg-slate-950 flex justify-between items-center px-6 h-16 w-full border-b border-outline-variant/10">
            <div class="flex items-center gap-8">
                <a href="/" class="font-manrope font-bold text-xl text-[#2d3435] dark:text-slate-100 no-underline">
                    Atheneum
                </a>
                <nav class="hidden md:flex items-center gap-6 font-manrope tracking-tight text-sm">
                    <a class="text-[#2F5597] font-bold border-b-2 border-[#2F5597] pb-1 cursor-pointer duration-200 ease-in-out" href="/">Collections</a>
                </nav>
            </div>
            <div class="flex items-center gap-4">
                <div id="authArea" class="flex items-center"></div>
                <div class="h-6 w-[1px] bg-outline-variant/30"></div>
                <div class="flex items-center bg-surface-container-low px-3 py-1 rounded-full text-xs font-label text-on-surface-variant">
                    <span class="w-2 h-2 rounded-full ${mode === 'edit' ? 'bg-tertiary' : 'bg-primary'} mr-2"></span>
                    ${modeLabel}
                </div>
                ${headerActions ? `
                    <div class="h-6 w-[1px] bg-outline-variant/30"></div>
                    ${headerActions}
                ` : ''}
            </div>
        </header>
        <div class="flex h-screen pt-16">
            <!-- SideNavBar -->
            <aside class="w-64 bg-[#ebeeef] dark:bg-slate-900 flex flex-col h-full py-8 gap-y-2 border-r-0">
                <div class="px-6 mb-8">
                    <h2 class="font-manrope font-semibold text-[#2d3435] dark:text-slate-200">Library</h2>
                    <p class="text-[10px] uppercase tracking-widest text-outline-variant">Personal Workspace</p>
                    ${mode === 'edit' ? `
                        <button onclick="addNewRoot()" data-auth-required style="display:none;" class="mt-6 w-full py-2 bg-gradient-to-br from-primary to-primary-dim text-on-primary rounded font-label text-xs tracking-wider uppercase flex items-center justify-center gap-2">
                            <span class="material-symbols-outlined text-sm">add</span>
                            New Root Node
                        </button>
                    ` : mode === 'index' ? `
                        <button onclick="createTree()" data-auth-required style="display:none;" class="mt-6 w-full py-2 bg-gradient-to-br from-primary to-primary-dim text-on-primary rounded font-label text-xs tracking-wider uppercase flex items-center justify-center gap-2">
                            <span class="material-symbols-outlined text-sm">add</span>
                            New Collection
                        </button>
                    ` : ''}
                </div>
                <nav class="flex-grow space-y-1">
                    <a href="/" class="flex items-center ${mode === 'index' ? 'text-[#2F5597] border-l-4 border-[#7a5a00] pl-4 bg-[#ffffff]/50' : 'text-[#2d3435] opacity-70 hover:opacity-100 pl-5 hover:bg-[#e4e9ea] dark:hover:bg-slate-800'} h-10 cursor-pointer transition-all duration-300 ease-out font-inter text-xs uppercase tracking-widest no-underline">
                        <span class="material-symbols-outlined mr-3 text-lg">account_tree</span>
                        Collections
                    </a>
                </nav>
            </aside>
            <!-- Main Content Area -->
            <main class="flex-grow overflow-y-auto bg-background">
                <div class="max-w-4xl mx-auto p-12">
                    <header class="mb-12">
                        ${options.badge ? `
                            <div class="flex items-center gap-3 mb-4">
                                <span class="px-2 py-0.5 ${badgeColor} text-[10px] font-bold rounded uppercase tracking-tighter">${options.badge}</span>
                            </div>
                        ` : ''}
                        ${options.title ? `
                            <h1 id="${options.titleId || 'pageTitle'}" class="text-5xl font-headline font-extrabold text-on-surface tracking-tight mb-4">${options.title}</h1>
                        ` : ''}
                        ${options.subtitle ? `
                            <p class="text-lg text-on-surface-variant leading-relaxed max-w-2xl font-body">${options.subtitle}</p>
                        ` : ''}
                    </header>
                    <div id="main-content"></div>
                </div>
            </main>
        </div>
    `;
}
