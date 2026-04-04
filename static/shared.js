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

// Render the common page layout
function renderLayout(options) {
    // options: { mode: 'view' | 'edit' | 'index', title: '', subtitle: '', badge: '', badgeColor: '', mainContent: '' }
    const mode = options.mode || 'index';
    const badgeColor = options.badgeColor || (mode === 'edit' ? 'bg-primary-container text-on-primary-container' : 'bg-tertiary-container text-on-tertiary-container');

    const topNavLinks = [
        { label: 'Focus', href: '#', active: true },
        { label: 'Publish', href: '#', active: false },
        { label: 'Archive', href: '#', active: false },
    ];

    const sidebarItems = [
        { label: 'Collections', icon: 'account_tree', active: true },
        { label: 'Recent', icon: 'schedule', active: false },
        { label: 'Favorites', icon: 'grade', active: false },
        { label: 'Trash', icon: 'delete', active: false },
    ];

    return `
        <!-- TopAppBar -->
        <header class="fixed top-0 z-50 bg-[#ffffff] dark:bg-slate-950 flex justify-between items-center px-6 h-16 w-full border-b border-outline-variant/10">
            <div class="flex items-center gap-8">
                <a href="/" class="font-manrope font-bold text-xl text-[#2d3435] dark:text-slate-100 no-underline">
                    Atheneum
                </a>
                <nav class="hidden md:flex items-center gap-6 font-manrope tracking-tight text-sm">
                    ${topNavLinks.map(l => `
                        <a class="${l.active ? 'text-[#2F5597] font-bold border-b-2 border-[#2F5597] pb-1' : 'text-[#adb3b4] hover:text-[#2d3435]'} cursor-pointer duration-200 ease-in-out ${!l.active ? 'hover:bg-[#ebeeef] dark:hover:bg-slate-800 transition-colors px-2 py-1 rounded' : ''}" href="${l.href}">${l.label}</a>
                    `).join('')}
                </nav>
            </div>
            <div class="flex items-center gap-4">
                <div class="flex items-center bg-surface-container-low px-3 py-1 rounded-full text-xs font-label text-on-surface-variant">
                    <span class="w-2 h-2 rounded-full ${mode === 'edit' ? 'bg-tertiary' : 'bg-primary'} mr-2"></span>
                    ${mode === 'edit' ? 'Edit Mode' : mode === 'view' ? 'View Mode' : 'Home'}
                </div>
                <div class="h-6 w-[1px] bg-outline-variant/30"></div>
                ${mode === 'edit' ? `
                    <button onclick="saveTree()" class="flex items-center gap-2 px-4 py-1.5 bg-surface-container-highest text-on-surface rounded font-label text-sm hover:bg-surface-container-high transition-colors">
                        <span class="material-symbols-outlined text-sm">save</span>
                        Save
                    </button>
                    <a id="viewLink" href="#" class="flex items-center gap-2 px-4 py-1.5 bg-primary text-on-primary rounded font-label text-sm hover:bg-primary-dim transition-colors no-underline">
                        <span class="material-symbols-outlined text-sm">visibility</span>
                        View
                    </a>
                ` : mode === 'view' ? `
                    <a id="editLink" href="#" class="flex items-center gap-2 px-4 py-1.5 bg-surface-container-highest text-on-surface rounded font-label text-sm hover:bg-surface-container-high transition-colors no-underline">
                        <span class="material-symbols-outlined text-sm">edit</span>
                        Edit
                    </a>
                ` : ''}
                <div class="flex items-center gap-3 ml-4">
                    <span class="material-symbols-outlined text-[#2F5597] cursor-pointer hover:bg-[#ebeeef] p-1.5 rounded transition-all">search</span>
                    <span class="material-symbols-outlined text-[#2F5597] cursor-pointer hover:bg-[#ebeeef] p-1.5 rounded transition-all">edit_note</span>
                    <span class="material-symbols-outlined text-[#2F5597] cursor-pointer hover:bg-[#ebeeef] p-1.5 rounded transition-all">visibility</span>
                </div>
            </div>
        </header>
        <div class="flex h-screen pt-16">
            <!-- SideNavBar -->
            <aside class="w-64 bg-[#ebeeef] dark:bg-slate-900 flex flex-col h-full py-8 gap-y-2 border-r-0">
                <div class="px-6 mb-8">
                    <h2 class="font-manrope font-semibold text-[#2d3435] dark:text-slate-200">Library</h2>
                    <p class="text-[10px] uppercase tracking-widest text-outline-variant">Personal Workspace</p>
                    ${mode === 'edit' ? `
                        <button onclick="addNewRoot()" class="mt-6 w-full py-2 bg-gradient-to-br from-primary to-primary-dim text-on-primary rounded font-label text-xs tracking-wider uppercase flex items-center justify-center gap-2">
                            <span class="material-symbols-outlined text-sm">add</span>
                            New Root Node
                        </button>
                    ` : mode === 'index' ? `
                        <button onclick="createTree()" class="mt-6 w-full py-2 bg-gradient-to-br from-primary to-primary-dim text-on-primary rounded font-label text-xs tracking-wider uppercase flex items-center justify-center gap-2">
                            <span class="material-symbols-outlined text-sm">add</span>
                            New Collection
                        </button>
                    ` : ''}
                </div>
                <nav class="flex-grow space-y-1">
                    ${sidebarItems.map(item => `
                        <div class="flex items-center ${item.active ? 'text-[#2F5597] border-l-4 border-[#7a5a00] pl-4 bg-[#ffffff]/50' : 'text-[#2d3435] opacity-70 hover:opacity-100 pl-5 hover:bg-[#e4e9ea] dark:hover:bg-slate-800'} h-10 cursor-pointer transition-all duration-300 ease-out font-inter text-xs uppercase tracking-widest">
                            <span class="material-symbols-outlined mr-3 text-lg">${item.icon}</span>
                            ${item.label}
                        </div>
                    `).join('')}
                </nav>
                <div class="mt-auto px-6 space-y-2 border-t border-outline-variant/10 pt-4">
                    <div class="flex items-center gap-3 text-on-surface-variant hover:text-on-surface cursor-pointer text-xs font-label">
                        <span class="material-symbols-outlined text-lg">settings</span>
                        Settings
                    </div>
                    <div class="flex items-center gap-3 text-on-surface-variant hover:text-on-surface cursor-pointer text-xs font-label">
                        <span class="material-symbols-outlined text-lg">help_outline</span>
                        Help
                    </div>
                </div>
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
