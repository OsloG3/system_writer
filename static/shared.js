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

// ---- Accounts and permissions ----

window.currentUser = null; // username of the logged-in user, or null
window.appUnlocked = false; // legacy alias for "logged in"
window.currentTree = null; // tree loaded by the current page (set by pages)

// Query the server for the current auth state, update UI and notify pages
async function refreshAuthState() {
    try {
        const res = await fetch("/api/auth");
        const data = await res.json();
        window.currentUser = data.loggedIn ? data.username : null;
    } catch (e) {
        window.currentUser = null;
    }
    window.appUnlocked = !!window.currentUser;
    applyAuthUI();
    document.dispatchEvent(new CustomEvent("authchange"));
}

// Render the login button or the user chip + logout in the header
function applyAuthUI() {
    const authArea = document.getElementById("authArea");
    if (authArea) {
        authArea.innerHTML = window.currentUser
            ? `<div class="flex items-center gap-1.5 sm:gap-2">
                    <span class="hidden md:flex items-center gap-1 px-2 py-1 bg-surface-container-low rounded-full text-xs font-label text-on-surface-variant">
                        <span class="material-symbols-outlined text-sm">person</span>
                        ${escapeHtml(window.currentUser)}
                    </span>
                    <button onclick="logoutNow()" title="Log out" class="flex items-center gap-1 sm:gap-2 px-2 sm:px-3 py-1.5 bg-surface-container text-on-surface-variant rounded font-label text-sm hover:bg-surface-container-high transition-colors">
                        <span class="material-symbols-outlined text-sm">logout</span>
                        <span class="hidden sm:inline">Log out</span>
                    </button>
               </div>`
            : `<button onclick="showAuthDialog('login')" title="Log in" class="flex items-center gap-1 sm:gap-2 px-2 sm:px-3 py-1.5 bg-tertiary-container/40 text-on-tertiary-container rounded font-label text-sm hover:bg-tertiary-container/70 transition-colors">
                    <span class="material-symbols-outlined text-sm">login</span>
                    <span class="hidden sm:inline">Log in</span>
               </button>`;
    }
}

// Show/hide per-tree action buttons once the tree and user are known
function applyTreePerms(tree) {
    const canEdit = treeCanEdit(tree);
    const owner = treeIsOwner(tree);
    document.querySelectorAll("[data-edit-required]").forEach((el) => {
        el.style.display = canEdit ? "" : "none";
    });
    document.querySelectorAll("[data-owner-required]").forEach((el) => {
        el.style.display = owner ? "" : "none";
    });
}

// The owner and the editors they added may edit; legacy trees with no
// owner may be edited by any logged-in user
function treeCanEdit(tree) {
    if (!tree || !window.currentUser) return false;
    if (!tree.owner) return true;
    if (tree.owner === window.currentUser) return true;
    return (tree.editors || []).includes(window.currentUser);
}

function treeIsOwner(tree) {
    if (!tree || !window.currentUser) return false;
    return !tree.owner || tree.owner === window.currentUser;
}

// ---- Login / register dialog ----

function showAuthDialog(tab) {
    closeAuthDialog();
    const mode = tab === "register" ? "register" : "login";
    const overlay = document.createElement("div");
    overlay.id = "authOverlay";
    overlay.className = "unlock-overlay";
    overlay.innerHTML = `
        <div class="unlock-card">
            <span class="material-symbols-outlined text-primary text-3xl mb-2">person</span>
            <div class="flex gap-2 mb-3 w-full">
                <button type="button" id="tabLogin" class="flex-1 py-1.5 rounded font-label text-sm transition-colors">Log in</button>
                <button type="button" id="tabRegister" class="flex-1 py-1.5 rounded font-label text-sm transition-colors">Create account</button>
            </div>
            <p id="authSubtitle" class="text-sm text-on-surface-variant mb-4"></p>
            <form id="authForm">
                <input id="authUser" type="text" autocomplete="username" placeholder="Username" class="unlock-input mb-2" />
                <input id="authPass" type="password" autocomplete="current-password" placeholder="Password" class="unlock-input" />
                <input id="authPass2" type="password" autocomplete="new-password" placeholder="Confirm password" class="unlock-input mt-2" style="display:none;" />
                <div id="authError" class="text-xs mt-2" style="color:#9f403d; display:none;"></div>
                <div class="flex justify-end gap-2 mt-4">
                    <button type="button" id="authCancel" class="px-4 py-2 bg-surface-container text-on-surface-variant rounded font-label text-sm hover:bg-surface-container-high transition-colors">Cancel</button>
                    <button type="submit" id="authSubmit" class="px-4 py-2 bg-primary text-on-primary rounded font-label text-sm hover:bg-primary-dim transition-colors"></button>
                </div>
            </form>
        </div>
    `;
    document.body.appendChild(overlay);

    let currentMode = mode;
    const userInput = overlay.querySelector("#authUser");
    const passInput = overlay.querySelector("#authPass");
    const pass2Input = overlay.querySelector("#authPass2");
    const errorEl = overlay.querySelector("#authError");

    function paintMode() {
        const register = currentMode === "register";
        overlay.querySelector("#tabLogin").className =
            "flex-1 py-1.5 rounded font-label text-sm transition-colors " +
            (register
                ? "bg-surface-container text-on-surface-variant"
                : "bg-primary text-on-primary");
        overlay.querySelector("#tabRegister").className =
            "flex-1 py-1.5 rounded font-label text-sm transition-colors " +
            (register
                ? "bg-primary text-on-primary"
                : "bg-surface-container text-on-surface-variant");
        overlay.querySelector("#authSubtitle").textContent = register
            ? "Create an account to build your own systems and edit ones you are added to."
            : "Log in to create systems and edit the ones you own or were added to.";
        pass2Input.style.display = register ? "" : "none";
        passInput.autocomplete = register ? "new-password" : "current-password";
        overlay.querySelector("#authSubmit").textContent = register
            ? "Create account"
            : "Log in";
        errorEl.style.display = "none";
    }
    overlay.querySelector("#tabLogin").onclick = () => {
        currentMode = "login";
        paintMode();
    };
    overlay.querySelector("#tabRegister").onclick = () => {
        currentMode = "register";
        paintMode();
    };
    paintMode();

    userInput.focus();
    overlay.querySelector("#authCancel").onclick = closeAuthDialog;
    overlay.addEventListener("click", (e) => {
        if (e.target === overlay) closeAuthDialog();
    });
    overlay.querySelector("#authForm").onsubmit = async (e) => {
        e.preventDefault();
        const username = userInput.value.trim();
        const password = passInput.value;
        if (currentMode === "register" && password !== pass2Input.value) {
            errorEl.textContent = "Passwords do not match";
            errorEl.style.display = "block";
            return;
        }
        const submit = overlay.querySelector("#authSubmit");
        submit.disabled = true;
        const res = await fetch(
            currentMode === "register" ? "/api/register" : "/api/login",
            {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ username: username, password: password }),
            },
        );
        submit.disabled = false;
        if (res.ok) {
            const data = await res.json();
            closeAuthDialog();
            showToast(
                currentMode === "register"
                    ? "Account created, welcome " + data.username
                    : "Logged in as " + data.username,
            );
            refreshAuthState();
        } else {
            let msg = currentMode === "register" ? "Registration failed" : "Incorrect username or password";
            try {
                const data = await res.json();
                if (data.error) msg = data.error;
            } catch (e) {}
            errorEl.textContent = msg;
            errorEl.style.display = "block";
            passInput.value = "";
            pass2Input.value = "";
            passInput.focus();
        }
    };
}

function closeAuthDialog() {
    const overlay = document.getElementById("authOverlay");
    if (overlay) overlay.remove();
}

// Backwards-compatible name used by older call sites
function showUnlockDialog() {
    showAuthDialog("login");
}

async function logoutNow() {
    await fetch("/api/logout", { method: "POST" });
    showToast("Logged out");
    refreshAuthState();
}

// ---- Copying a system ----

// Duplicate the tree loaded on the current page as a new system owned by
// the current user, then open it in the editor
async function copyCurrentTree() {
    const tree = window.currentTree;
    if (!tree) return;
    if (!window.currentUser) {
        showAuthDialog("login");
        return;
    }
    const res = await fetch(`/api/tree/${tree.id}/copy`, { method: "POST" });
    if (res.ok) {
        const data = await res.json();
        window.location.href = "/edit/" + data.id;
    } else if (res.status === 401) {
        showAuthDialog("login");
    } else {
        showToast("Could not copy system", true);
    }
}

// ---- Sharing (managing editors) ----

// Open the sharing dialog for the tree loaded on the current page
function shareCurrentTree() {
    if (window.currentTree) showShareDialog(window.currentTree);
}

async function showShareDialog(tree) {
    closeShareDialog();
    let allUsers = [];
    try {
        const res = await fetch("/api/users");
        if (res.ok) allUsers = (await res.json()).users || [];
    } catch (e) {}

    const overlay = document.createElement("div");
    overlay.id = "shareOverlay";
    overlay.className = "unlock-overlay";
    overlay.innerHTML = `
        <div class="unlock-card">
            <span class="material-symbols-outlined text-primary text-3xl mb-2">group</span>
            <h2 class="font-headline font-bold text-on-surface text-lg mb-1">Sharing</h2>
            <p class="text-sm text-on-surface-variant mb-3">Editors can change this system, but only the owner can delete it or manage editors.</p>
            <div class="text-xs text-on-surface-variant mb-3">
                Owner: <span class="font-semibold">${tree.owner ? escapeHtml(tree.owner) : "<em>unclaimed (legacy system)</em>"}</span>
            </div>
            <div id="editorList" class="w-full space-y-1.5 mb-3"></div>
            <form id="addEditorForm" class="w-full">
                <div class="flex gap-2">
                    <input id="editorUser" type="text" list="editorUserList" placeholder="Username to add" autocomplete="off" class="unlock-input" />
                    <datalist id="editorUserList"></datalist>
                    <button type="submit" class="shrink-0 px-3 py-2 bg-primary text-on-primary rounded font-label text-sm hover:bg-primary-dim transition-colors">Add</button>
                </div>
                <div id="shareError" class="text-xs mt-2" style="color:#9f403d; display:none;"></div>
            </form>
            <div class="flex justify-end mt-4 w-full">
                <button type="button" id="shareClose" class="px-4 py-2 bg-surface-container text-on-surface-variant rounded font-label text-sm hover:bg-surface-container-high transition-colors">Close</button>
            </div>
        </div>
    `;
    document.body.appendChild(overlay);

    const listEl = overlay.querySelector("#editorList");
    const errorEl = overlay.querySelector("#shareError");
    const userInput = overlay.querySelector("#editorUser");
    const datalist = overlay.querySelector("#editorUserList");

    function renderEditors() {
        const editors = tree.editors || [];
        listEl.innerHTML = editors.length
            ? editors
                  .map(
                      (u) => `
                <div class="flex items-center justify-between px-3 py-2 bg-surface-container-low rounded">
                    <span class="flex items-center gap-2 text-sm text-on-surface">
                        <span class="material-symbols-outlined text-base text-primary">edit</span>
                        ${escapeHtml(u)}
                    </span>
                    <button type="button" data-remove="${escapeHtml(u)}" title="Remove editor" class="px-2 py-1 bg-error-container/40 text-on-error-container rounded text-xs font-label hover:bg-error-container transition-colors">Remove</button>
                </div>`,
                  )
                  .join("")
            : `<p class="text-sm text-on-surface-variant italic">No editors yet.</p>`;
        listEl.querySelectorAll("[data-remove]").forEach((btn) => {
            btn.onclick = () => changeEditor(btn.dataset.remove, "remove");
        });
        datalist.innerHTML = allUsers
            .filter((u) => u !== tree.owner && !editors.includes(u))
            .map((u) => `<option value="${escapeHtml(u)}"></option>`)
            .join("");
    }

    async function changeEditor(username, action) {
        errorEl.style.display = "none";
        const res = await fetch(`/api/tree/${tree.id}/editors`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ username: username, action: action }),
        });
        if (res.ok) {
            const data = await res.json();
            tree.editors = data.editors || [];
            renderEditors();
            showToast(
                action === "add"
                    ? `${username} can now edit`
                    : `${username} removed`,
            );
            document.dispatchEvent(new CustomEvent("editorschange"));
        } else {
            let msg = "Could not update editors";
            try {
                const data = await res.json();
                if (data.error) msg = data.error;
            } catch (e) {}
            errorEl.textContent = msg;
            errorEl.style.display = "block";
        }
    }

    overlay.querySelector("#addEditorForm").onsubmit = (e) => {
        e.preventDefault();
        const username = userInput.value.trim().toLowerCase();
        if (!username) return;
        userInput.value = "";
        changeEditor(username, "add");
    };
    overlay.querySelector("#shareClose").onclick = closeShareDialog;
    overlay.addEventListener("click", (e) => {
        if (e.target === overlay) closeShareDialog();
    });

    renderEditors();
    userInput.focus();
}

function closeShareDialog() {
    const overlay = document.getElementById("shareOverlay");
    if (overlay) overlay.remove();
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
        showToast("Log in first", true);
        showAuthDialog("login");
    } else if (res.status === 403) {
        showToast("Only the owner can delete this system", true);
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

    const btnGhost = "flex items-center gap-1 sm:gap-2 px-2 sm:px-4 py-1.5 bg-surface-container-highest text-on-surface rounded font-label text-sm hover:bg-surface-container-high transition-colors no-underline";
    const btnPrimary = "flex items-center gap-1 sm:gap-2 px-2 sm:px-4 py-1.5 bg-primary text-on-primary rounded font-label text-sm hover:bg-primary-dim transition-colors no-underline";
    const btnDanger = "flex items-center gap-1 sm:gap-2 px-2 sm:px-3 py-1.5 bg-error-container/30 text-on-error-container rounded font-label text-sm hover:bg-error-container/60 transition-colors";
    const deleteBtn = `
        <button onclick="deleteCurrentTree()" title="Delete system" data-owner-required style="display:none;" class="${btnDanger}">
            <span class="material-symbols-outlined text-sm">delete</span>
            <span class="hidden sm:inline">Delete</span>
        </button>
    `;
    const shareBtn = `
        <button onclick="shareCurrentTree()" title="Manage editors" data-owner-required style="display:none;" class="${btnGhost}">
            <span class="material-symbols-outlined text-sm">group</span>
            <span class="hidden sm:inline">Share</span>
        </button>
    `;

    let headerActions = '';
    if (mode === 'edit') {
        headerActions = `
            <button onclick="saveTree()" data-edit-required style="display:none;" class="${btnGhost}">
                <span class="material-symbols-outlined text-sm">save</span>
                <span class="hidden sm:inline">Save</span>
            </button>
            ${shareBtn}
            <a id="viewLink" href="#" class="${btnGhost}">
                <span class="material-symbols-outlined text-sm">visibility</span>
                <span class="hidden sm:inline">View</span>
            </a>
            <a id="practiceLink" href="#" class="${btnPrimary}">
                <span class="material-symbols-outlined text-sm">school</span>
                <span class="hidden sm:inline">Practice</span>
            </a>
            ${deleteBtn}
        `;
    } else if (mode === 'view') {
        headerActions = `
            <a id="practiceLink" href="#" class="${btnPrimary}">
                <span class="material-symbols-outlined text-sm">school</span>
                <span class="hidden sm:inline">Practice</span>
            </a>
            <button onclick="copyCurrentTree()" title="Create your own copy of this system" class="${btnGhost}">
                <span class="material-symbols-outlined text-sm">content_copy</span>
                <span class="hidden sm:inline">Copy</span>
            </button>
            <a id="editLink" href="#" data-edit-required style="display:none;" class="${btnGhost}">
                <span class="material-symbols-outlined text-sm">edit</span>
                <span class="hidden sm:inline">Edit</span>
            </a>
            ${shareBtn}
            ${deleteBtn}
        `;
    } else if (mode === 'practice') {
        headerActions = `
            <a id="viewLink" href="#" class="${btnGhost}">
                <span class="material-symbols-outlined text-sm">visibility</span>
                <span class="hidden sm:inline">View</span>
            </a>
            <a id="editLink" href="#" data-edit-required style="display:none;" class="${btnGhost}">
                <span class="material-symbols-outlined text-sm">edit</span>
                <span class="hidden sm:inline">Edit</span>
            </a>
            ${shareBtn}
            ${deleteBtn}
        `;
    }

    return `
        <!-- TopAppBar -->
        <header class="fixed top-0 z-50 bg-[#ffffff] dark:bg-slate-950 flex justify-between items-center px-3 sm:px-6 h-16 w-full border-b border-outline-variant/10">
            <div class="flex items-center gap-2 sm:gap-8 min-w-0">
                <button id="menuBtn" onclick="toggleSidebar()" aria-label="Open menu" title="Menu" class="md:hidden flex items-center justify-center w-9 h-9 shrink-0 rounded text-on-surface-variant hover:bg-surface-container transition-colors">
                    <span class="material-symbols-outlined">menu</span>
                </button>
                <a href="/" class="font-manrope font-bold text-lg sm:text-xl text-[#2d3435] dark:text-slate-100 no-underline truncate">
                    Atheneum
                </a>
                <nav class="hidden md:flex items-center gap-6 font-manrope tracking-tight text-sm">
                    <a class="text-[#2F5597] font-bold border-b-2 border-[#2F5597] pb-1 cursor-pointer duration-200 ease-in-out" href="/">Collections</a>
                </nav>
            </div>
            <div class="flex items-center gap-1.5 sm:gap-4 shrink-0">
                <div id="authArea" class="flex items-center"></div>
                <div class="hidden lg:flex items-center bg-surface-container-low px-3 py-1 rounded-full text-xs font-label text-on-surface-variant">
                    <span class="w-2 h-2 rounded-full ${mode === 'edit' ? 'bg-tertiary' : 'bg-primary'} mr-2"></span>
                    ${modeLabel}
                </div>
                ${headerActions ? `
                    <div class="hidden sm:block h-6 w-[1px] bg-outline-variant/30"></div>
                    <div class="flex items-center gap-1.5 sm:gap-3">
                        ${headerActions}
                    </div>
                ` : ''}
            </div>
        </header>
        <div class="app-shell flex pt-16">
            <!-- SideNavBar (slide-in drawer on small screens) -->
            <div id="sidebarBackdrop" onclick="closeSidebar()"></div>
            <aside id="sidebar" class="w-64 bg-[#ebeeef] dark:bg-slate-900 flex flex-col h-full py-8 gap-y-2 border-r-0">
                <div class="px-6 mb-8">
                    <h2 class="font-manrope font-semibold text-[#2d3435] dark:text-slate-200">Library</h2>
                    <p class="text-[10px] uppercase tracking-widest text-outline-variant">Personal Workspace</p>
                    ${mode === 'edit' ? `
                        <button onclick="addNewRoot(); closeSidebar();" data-edit-required style="display:none;" class="mt-6 w-full py-2 bg-gradient-to-br from-primary to-primary-dim text-on-primary rounded font-label text-xs tracking-wider uppercase flex items-center justify-center gap-2">
                            <span class="material-symbols-outlined text-sm">add</span>
                            New Root Node
                        </button>
                    ` : mode === 'index' ? `
                        <button onclick="createTree(); closeSidebar();" class="mt-6 w-full py-2 bg-gradient-to-br from-primary to-primary-dim text-on-primary rounded font-label text-xs tracking-wider uppercase flex items-center justify-center gap-2">
                            <span class="material-symbols-outlined text-sm">add</span>
                            New Collection
                        </button>
                    ` : ''}
                </div>
                <nav class="flex-grow space-y-1">
                    <a href="/" onclick="closeSidebar()" class="flex items-center ${mode === 'index' ? 'text-[#2F5597] border-l-4 border-[#7a5a00] pl-4 bg-[#ffffff]/50' : 'text-[#2d3435] opacity-70 hover:opacity-100 pl-5 hover:bg-[#e4e9ea] dark:hover:bg-slate-800'} h-10 cursor-pointer transition-all duration-300 ease-out font-inter text-xs uppercase tracking-widest no-underline">
                        <span class="material-symbols-outlined mr-3 text-lg">account_tree</span>
                        Collections
                    </a>
                </nav>
            </aside>
            <!-- Main Content Area -->
            <main class="flex-grow overflow-y-auto bg-background min-w-0">
                <div class="max-w-4xl mx-auto p-4 sm:p-6 md:p-12">
                    <header class="mb-8 md:mb-12">
                        ${options.badge ? `
                            <div class="flex items-center gap-3 mb-4">
                                <span class="px-2 py-0.5 ${badgeColor} text-[10px] font-bold rounded uppercase tracking-tighter">${options.badge}</span>
                            </div>
                        ` : ''}
                        ${options.title ? `
                            <h1 id="${options.titleId || 'pageTitle'}" class="text-3xl sm:text-4xl md:text-5xl font-headline font-extrabold text-on-surface tracking-tight mb-4">${options.title}</h1>
                        ` : ''}
                        ${options.subtitle ? `
                            <p class="text-base md:text-lg text-on-surface-variant leading-relaxed max-w-2xl font-body">${options.subtitle}</p>
                        ` : ''}
                    </header>
                    <div id="main-content"></div>
                </div>
            </main>
        </div>
    `;
}

// ---- Mobile sidebar drawer ----

function toggleSidebar(force) {
    const sidebar = document.getElementById("sidebar");
    if (!sidebar) return;
    const open = typeof force === "boolean" ? force : !sidebar.classList.contains("open");
    sidebar.classList.toggle("open", open);
    const backdrop = document.getElementById("sidebarBackdrop");
    if (backdrop) backdrop.classList.toggle("show", open);
    const btn = document.getElementById("menuBtn");
    if (btn) btn.setAttribute("aria-expanded", open ? "true" : "false");
}

function closeSidebar() {
    toggleSidebar(false);
}

document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeSidebar();
});

window.addEventListener("resize", () => {
    if (window.innerWidth >= 768) closeSidebar();
});
