<script lang="ts">
  import { onMount, tick } from 'svelte';
  import { design as persistedDesign } from '$lib/design/projected-design';
  import { designBenchVersion, projectName } from '$lib/design-bench';
  import { renderDiagram, renderSyntax } from '$lib/renderers';
  import '$lib/design/projected-design.css';

  let design = structuredClone(persistedDesign);
  const persistedJson = JSON.stringify(persistedDesign);
  let themeId = design.defaultTheme;
  let appearance: 'light' | 'dark' = design.defaultAppearance === 'dark' || (design.defaultAppearance === 'system' && typeof window !== 'undefined' && window.matchMedia('(prefers-color-scheme: dark)').matches) ? 'dark' : 'light';
  let pendingOperations: Array<Record<string, string>> = [];
  let saveMessage = '';
  let authorityRevision = '';
  let editingColor = '';
  let draftColor = '';
  let colorInput: HTMLInputElement;
  let editingValue = '';
  let draftValue = '';
  let valueInput: HTMLInputElement;
  let editingShadow = '';
  let draftShadow: { x: number; y: number; blur: number; spread: number } = { x: 0, y: 0, blur: 0, spread: 0 };
  const editingEnabled = import.meta.env.DEV;
  let copied = '';
  let inputValue = 'Editable specimen text';
  let selectValue = 'One';
  let checked = false;
  let buttonMessage = '';
  let syntaxHtml = '';
  let diagramHtml = '';
  let diagramElement: HTMLDivElement;
  const modes = ['light', 'dark'] as const;
  $: theme = (design.themes.find((candidate) => candidate.id === themeId) ?? design.themes[0]) as any;
  $: family = theme as any;
  $: supportedAppearances = Object.keys(theme.appearances) as Array<'light' | 'dark'>;
  $: if (!supportedAppearances.includes(appearance)) appearance = supportedAppearances[0] ?? 'light';
  $: active = theme.appearances[appearance] as any;
  $: if (hasRendererSpecimens && active) renderSyntax(active).then((html) => { syntaxHtml = html; });
  $: if (hasRendererSpecimens && active && diagramElement) renderDiagram(active, `design-bench-diagram-${themeId}-${appearance}`).then((svg) => { diagramHtml = svg; });
  // Renderer examples are an explicit theme capability: editorial-only themes
  // should not imply syntax, Mermaid, or Excalidraw support.
  $: hasRendererSpecimens = themeId === 'field-notes';
  $: dirty = JSON.stringify(design) !== persistedJson;
  $: if (typeof document !== 'undefined') {
    document.documentElement.dataset.theme = themeId;
    document.documentElement.dataset.appearance = appearance;
  }
  onMount(() => {
    if (design.defaultAppearance === 'system' && matchMedia('(prefers-color-scheme: dark)').matches) appearance = 'dark';
    if (editingEnabled) fetch(`/${['__design-bench', 'edit'].join('/')}`).then((response) => response.json()).then((payload) => { authorityRevision = payload.revision ?? ''; });
    return () => { diagramHtml = ''; syntaxHtml = ''; };
  });

  const entries = <T extends object>(value: T) => Object.entries(value);
  const token = (name: string) => `var(--color-${name.replaceAll('.', '-')})`;
  const humanize = (name: string) => name
    .replace(/([a-z0-9])([A-Z])/g, '$1 $2')
    .replace(/[_-]+/g, ' ')
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
  const fontName = (id: string) => design.fonts.find((font) => font.id === id)?.name ?? humanize(id);
  const readableForeground = (value: string) => {
    const hex = value.replace('#', '').slice(0, 6);
    if (hex.length !== 6) return '#111111';
    const channels = [0, 2, 4].map((offset) => parseInt(hex.slice(offset, offset + 2), 16) / 255).map((channel) => channel <= .03928 ? channel / 12.92 : Math.pow((channel + .055) / 1.055, 2.4));
    return .2126 * channels[0] + .7152 * channels[1] + .0722 * channels[2] > .48 ? '#111111' : '#FFFFFF';
  };
  const isUnset = (value: unknown) => value == null || value === '';
  const copyValue = async (key: string, value: unknown) => {
    const text = String(value);
    let copiedSuccessfully = false;
    try {
      if (typeof navigator !== 'undefined' && navigator.clipboard) {
        await navigator.clipboard.writeText(text);
        copiedSuccessfully = true;
      }
    } catch {
      copiedSuccessfully = false;
    }
    if (!copiedSuccessfully && typeof document !== 'undefined') {
      const input = document.createElement('textarea');
      input.value = text;
      input.setAttribute('readonly', '');
      input.style.position = 'fixed';
      input.style.opacity = '0';
      document.body.appendChild(input);
      input.select();
      copiedSuccessfully = document.execCommand('copy');
      input.remove();
    }
    copied = key;
    window.setTimeout(() => { if (copied === key) copied = ''; }, 1200);
  };
  const editColor = (role: string, value: string, custom = false) => {
    const next = structuredClone(design);
    const target: any = next.themes.find((candidate: any) => candidate.id === themeId);
    if (!target?.appearances[appearance]) return;
    const colors = custom ? target.appearances[appearance].customColors : target.appearances[appearance].colors;
    colors[role] = value;
    if (typeof document !== 'undefined') document.documentElement.style.setProperty(`--color-${role.replaceAll('.', '-')}`, value);
    design = next;
    pendingOperations = [...pendingOperations.filter((operation) => !(operation.theme === themeId && operation.appearance === appearance && operation.role === role)), { op: custom ? 'set_custom_color' : 'set_semantic_color', theme: themeId, appearance, role, value }];
  };
  const queueOperation = (operation: Record<string, string>) => { pendingOperations = [...pendingOperations, operation]; };
  const cloneAppearance = (source: any) => structuredClone(source);
  const createFamily = () => {
    const id = window.prompt('New theme identifier (lowercase, hyphenated):', 'new-theme')?.trim();
    if (!id || design.themes.some((candidate) => candidate.id === id)) return;
    const source = design.themes.find((candidate) => candidate.id === themeId) ?? theme;
    const appearances = structuredClone(source.appearances);
    design = { ...design, themes: [...design.themes, { ...structuredClone(source), id, appearances }] };
    queueOperation({ op: 'create_theme', theme: id, from: themeId, appearances: Object.keys(appearances).join(',') });
    themeId = id;
    appearance = supportedAppearances[0] ?? 'light';
  };
  const renameFamily = () => {
    const next = window.prompt('Rename theme identifier:', themeId)?.trim();
    if (!next || next === themeId || design.themes.some((candidate) => candidate.id === next)) return;
    const previous = themeId;
    design = { ...design, defaultTheme: design.defaultTheme === previous ? next : design.defaultTheme, themes: design.themes.map((candidate) => candidate.id === previous ? { ...candidate, id: next } : candidate) };
    queueOperation({ op: 'rename_theme', from: previous, to: next });
    themeId = next;
  };
  const removeFamily = () => {
    if (design.themes.length <= 1 || design.defaultTheme === themeId || !window.confirm(`Remove ${humanize(themeId)}?`)) return;
    design = { ...design, themes: design.themes.filter((candidate) => candidate.id !== themeId) };
    queueOperation({ op: 'remove_theme', theme: themeId });
    themeId = design.defaultTheme;
  };
  const setDefaultTheme = () => { design = { ...design, defaultTheme: themeId }; queueOperation({ op: 'set_default_theme', theme: themeId }); };
  const addAppearance = (mode: 'light' | 'dark') => {
    if (theme.appearances[mode]) return;
    const from = supportedAppearances[0];
    const updated = { ...theme, appearances: { ...theme.appearances, [mode]: cloneAppearance(theme.appearances[from]) } };
    design = { ...design, themes: design.themes.map((candidate) => candidate.id === themeId ? updated : candidate) };
    queueOperation({ op: 'add_appearance', theme: themeId, appearance: mode, from });
  };
  const removeAppearance = (mode: 'light' | 'dark') => {
    if (supportedAppearances.length <= 1 || !window.confirm(`Remove ${humanize(mode)} appearance values?`)) return;
    const appearances = { ...theme.appearances }; delete appearances[mode];
    design = { ...design, themes: design.themes.map((candidate) => candidate.id === themeId ? { ...candidate, appearances } : candidate) };
    queueOperation({ op: 'remove_appearance', theme: themeId, appearance: mode });
  };
  const setFamilyValue = (section: string, role: string, field: string, value: number | string) => {
    const next = structuredClone(design);
    const target: any = next.themes.find((candidate: any) => candidate.id === themeId);
    if (!target) return;
    const normalized = typeof value === 'number' ? value : Number(value);
    const nextValue = typeof value === 'number' ? value : (Number.isFinite(normalized) ? normalized : value);
    if (section === 'typography') target.typography[role][field === 'line_height' ? 'lineHeight' : field] = nextValue;
    else if (section === 'shadows') target.shadows[role][field] = nextValue;
    else target[section][role] = nextValue;
    design = next;
    pendingOperations = [...pendingOperations.filter((operation) => !(operation.op === 'set_theme_value' && operation.theme === themeId && operation.section === section && operation.role === role && operation.field === field)), { op: 'set_theme_value', theme: themeId, section, role, field, value: String(nextValue) }];
  };
  const beginValueEdit = (key: string, value: number) => {
    editingValue = key;
    draftValue = String(value);
    tick().then(() => { valueInput?.focus(); valueInput?.select(); });
  };
  const valuePreview = (key: string, value: number) => editingValue === key && draftValue.trim() !== '' && Number.isFinite(Number(draftValue)) ? Number(draftValue) : value;
  const commitValueEdit = (section: string, role: string, field: string) => {
    const value = Number(draftValue);
    if (Number.isFinite(value)) setFamilyValue(section, role, field, value);
    editingValue = '';
    draftValue = '';
  };
  const cancelValueEdit = () => { editingValue = ''; draftValue = ''; };
  const beginShadowEdit = (role: string, value: any) => {
    editingShadow = role;
    draftShadow = { x: value.x, y: value.y, blur: value.blur, spread: value.spread };
  };
  const updateShadowField = (role: string, field: 'x' | 'y' | 'blur' | 'spread', value: string) => {
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return;
    draftShadow = { ...draftShadow, [field]: numeric };
    setFamilyValue('shadows', role, field, numeric);
  };
  const closeShadowEdit = () => { editingShadow = ''; };
  const changeFontFamily = (role: string, value: string) => setFamilyValue('typography', role, 'font_family', value);
  const addCustomRole = () => {
    const role = window.prompt('New custom semantic role:', 'custom.role')?.trim();
    const value = window.prompt('Color value (#RRGGBB or #RRGGBBAA):', '#808080')?.trim();
    if (!role || !value || !/^#(?:[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$/.test(value)) return;
    const next = structuredClone(design);
    const target: any = next.themes.find((candidate: any) => candidate.id === themeId);
    if (!target) return;
    for (const mode of Object.keys(target.appearances)) target.appearances[mode].customColors[role] = value;
    design = next;
    for (const mode of Object.keys(target.appearances)) queueOperation({ op: 'set_custom_color', theme: themeId, appearance: mode, role, value });
  };
  const removeCustomRole = (role: string) => {
    if (!window.confirm(`Remove ${role} from this family?`)) return;
    const next = structuredClone(design);
    const target: any = next.themes.find((candidate: any) => candidate.id === themeId);
    if (!target) return;
    for (const mode of Object.keys(target.appearances)) delete target.appearances[mode].customColors[role];
    design = next;
    queueOperation({ op: 'remove_custom_color', theme: themeId, role });
  };
  const beginColorEdit = (role: string, value: string | null | undefined, custom = false) => {
    editingColor = `${custom ? 'custom:' : 'color:'}${role}`;
    draftColor = value ?? '';
    tick().then(() => { colorInput?.focus(); colorInput?.select(); });
  };
  const commitColorEdit = (role: string, custom = false) => {
    if (/^#(?:[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$/.test(draftColor.trim())) editColor(role, draftColor.trim(), custom);
    editingColor = '';
  };
  const cancelColorEdit = () => { editingColor = ''; draftColor = ''; };
  const discardDraft = () => { design = structuredClone(persistedDesign); pendingOperations = []; saveMessage = ''; };
  const saveDraft = async () => {
    saveMessage = '';
    if (!editingEnabled) { saveMessage = 'Editing is available only in local development.'; return; }
    const response = await fetch(`/${['__design-bench', 'edit'].join('/')}`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ revision: authorityRevision, operations: pendingOperations }) });
    const payload = await response.json();
    saveMessage = response.ok ? 'Saved design authority and refreshed projections.' : (payload.message ?? payload.detail ?? 'Design save failed.');
    if (response.ok) window.location.reload();
  };
</script>

<svelte:head><title>Design Bench</title><meta name="description" content="Project design-system specimens" /></svelte:head>

<header class="toolbar">
  <div class="identity"><p class="eyebrow">{projectName}</p><h1>Design Bench <small class="bench-version">v{designBenchVersion}</small></h1></div>
  <div class="toolbar-controls">
    <div class="theme-control"><span class="toolbar-label">Theme</span><div class="theme-control-shell"><label class="theme-select"><select bind:value={themeId}>{#each design.themes as item}<option value={item.id}>{humanize(item.id)}</option>{/each}</select></label><details class="theme-menu">
        <summary aria-label="Manage themes" title="Manage themes">•••</summary>
        <div class="theme-menu-panel">
          <div class="menu-heading"><strong>{humanize(theme.id)}</strong>{#if design.defaultTheme === themeId}<span>Default</span>{/if}</div>
          <div class="management-row"><button type="button" onclick={createFamily}>New theme</button><button class="secondary" type="button" onclick={renameFamily}>Rename</button></div>
          <div class="management-row"><button class="secondary" type="button" onclick={setDefaultTheme} disabled={design.defaultTheme === themeId}>Make default</button><button class="danger" type="button" onclick={removeFamily} disabled={design.themes.length <= 1 || design.defaultTheme === themeId}>Remove</button></div>
          <div class="appearance-list"><p class="menu-label">Appearances</p>{#each modes as mode}<div class="appearance-row"><span>{humanize(mode)}</span>{#if theme.appearances[mode]}<span class="supported">Supported</span><button class="danger text-button" type="button" onclick={() => removeAppearance(mode)} disabled={supportedAppearances.length <= 1}>Remove</button>{:else}<button class="secondary text-button" type="button" onclick={() => addAppearance(mode)}>+ Add</button>{/if}</div>{/each}</div>
        </div>
      </details></div></div>
    <fieldset class="appearance-control"><legend>Appearance</legend><div class="segments">{#each supportedAppearances as mode}<label><input type="radio" bind:group={appearance} value={mode} aria-label={humanize(mode)} /><span aria-hidden="true">{#if mode === 'light'}<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="3.5" /><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41" /></svg>{:else}<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M20.5 14.5A8.5 8.5 0 0 1 9.5 3.5a8.6 8.6 0 1 0 11 11Z" /></svg>{/if}</span></label>{/each}</div></fieldset>
  </div>
  <div class="save-controls">{#if dirty}<span class="dirty" aria-live="polite">Unsaved changes</span><button type="button" onclick={saveDraft}>Save</button><button class="secondary" type="button" onclick={discardDraft}>Discard</button>{/if}</div>
</header>

<main>
  <section class="context"><span><strong>{projectName}</strong> · project-owned design authority</span><span>Default theme: <strong>{humanize(design.defaultTheme)}</strong> · Default appearance: <strong>{design.defaultAppearance}</strong></span></section>

  <section class="mobile-shell-specimen" aria-labelledby="mobile-shell-heading">
    <div class="section-heading"><div><p class="eyebrow">Reusable application presentation</p><h2 id="mobile-shell-heading">Mobile application shell</h2></div><span class="specimen-badge">{humanize(appearance)} appearance</span></div>
    <p class="section-note">A generic shell specimen exposes the semantic surfaces used by project-owned mobile applications. Change appearance above to pressure both light and dark projections without importing a source palette.</p>
    <div class="mobile-shell-frame">
      <header class="mobile-shell-header"><span class="eyebrow">Application header</span><span class="mobile-shell-status">{humanize(themeId)}</span></header>
      <div class="mobile-shell-body"><div><p class="eyebrow">Primary content</p><h3>Responsive application surface</h3><p>Content remains project-owned while the shell supplies viewport, safe-area, and layering seams.</p></div><div class="mobile-shell-overlay" aria-label="Overlay control">Overlay / elevated control</div></div>
      <footer class="mobile-shell-footer"><span>Inactive</span><button type="button" aria-label="Active shell control">Active control</button><span>Safe-area region</span></footer>
    </div>
  </section>

  <section><h2>UI Controls / Components</h2><p>Generic controls rendered from the active projected design. Interactions are local specimen state only.</p><div class="specimen-grid"><article class="surface"><p class="eyebrow">Buttons</p><div class="control-row"><button type="button" onclick={() => buttonMessage = 'Primary button activated'}>Primary</button><button class="secondary" type="button" onclick={() => buttonMessage = 'Secondary button activated'}>Secondary</button><button type="button" disabled>Disabled</button></div>{#if buttonMessage}<p class="message" role="status">{buttonMessage}</p>{/if}</article><form class="surface" onsubmit={(event) => event.preventDefault()}><p class="eyebrow">Form controls</p><label>Text input<input bind:value={inputValue} aria-label="Text input specimen" /></label><label>Select<select bind:value={selectValue} aria-label="Select specimen"><option>One</option><option>Two</option><option>Three</option></select></label><label class="checkbox-row"><input type="checkbox" bind:checked={checked} aria-label="Checkbox specimen" /> <span>Checkbox ({checked ? 'checked' : 'unchecked'})</span></label></form><article class="surface"><p class="eyebrow">Container and message</p><p>A card demonstrates surface, border, spacing, radius, and shadow semantics.</p><p class="message" role="status">Semantic messaging uses declared feedback colors.</p></article></div></section>

  <section class="content-specimens"><h2>Editorial / Content</h2><p class="section-note">A framework-neutral content specimen exposes reading hierarchy, links, inline code, code, media, diagrams, and drawing surfaces without importing an application renderer.</p><article class="editorial-specimen surface"><p class="eyebrow">Editorial hierarchy</p><h3>A considered title for a long-form surface</h3><p class="editorial-lead">A short introduction shows how the display and body roles carry a calm reading rhythm across a theme and appearance.</p><p>Body copy tests measure, line height, muted metadata, and <a href="/" onclick={(event) => event.preventDefault()}>inline links</a>. It also includes <code>inline code</code> so technical content remains legible inside prose.</p><blockquote>Evidence should be visible in the surface that helps someone evaluate it.</blockquote><ul><li>Lists retain readable spacing.</li><li>Secondary content remains subordinate.</li></ul><figure class="editorial-figure"><div class="media-placeholder" role="img" aria-label="Representative editorial media surface">Image / media</div><figcaption>A caption demonstrates authored media context.</figcaption></figure><pre class="syntax-specimen" aria-label="Representative Elixir syntax specimen"><code><span class="syntax-keyword">defmodule</span> <span class="syntax-name">Bench</span><span class="syntax-punctuation">.</span><span class="syntax-type">Specimen</span> <span class="syntax-keyword">do</span>
  <span class="syntax-keyword">def</span> <span class="syntax-name">signal</span><span class="syntax-punctuation">,</span> <span class="syntax-keyword">do:</span> <span class="syntax-string">:visible</span>
<span class="syntax-keyword">end</span></code></pre><div class="renderer-grid"><figure class="renderer-specimen"><div class="diagram-placeholder" role="img" aria-label="Representative diagram surface">Diagram / flow</div><figcaption>Diagram surface</figcaption></figure><figure class="renderer-specimen"><div class="drawing-placeholder" role="img" aria-label="Representative drawing canvas surface">Drawing / canvas</div><figcaption>Drawing surface</figcaption></figure></div></article></section>

  {#if hasRendererSpecimens}
    <section class="renderer-examples">
      <h2>Renderer examples</h2>
      <p class="section-note">Real syntax and diagram renderers consume the selected appearance. The drawing remains a source-derived representative specimen because authored canvas geometry stays renderer-owned.</p>
      <div class="syntax-specimen" aria-label="Actual Shiki Elixir syntax specimen">{#if syntaxHtml}{@html syntaxHtml}{:else}<p>Rendering syntax…</p>{/if}</div>
      <div class="renderer-grid">
        <figure class="renderer-specimen"><div class="mermaid-specimen" bind:this={diagramElement} role="img" aria-label="Actual Mermaid diagram">{#if diagramHtml}{@html diagramHtml}{:else}<p>Rendering diagram…</p>{/if}</div><pre class="renderer-source">flowchart LR; Intent --&gt; Architecture --&gt; Implementation --&gt; System</pre><figcaption>Mermaid diagram</figcaption></figure>
        <figure class="renderer-specimen"><div class="excalidraw-specimen" role="img" aria-label="Representative Excalidraw drawing"><svg viewBox="0 0 360 120" aria-hidden="true"><path class="drawing-line" d="M86 60h72M204 60h70"/><rect class="drawing-shape" x="16" y="36" width="70" height="48"/><path class="drawing-shape" d="M158 36l24 24-24 24-24-24z"/><rect class="drawing-shape" x="274" y="36" width="70" height="48"/><text x="51" y="65">Idea</text><text x="182" y="65">Shape</text><text x="309" y="65">Share</text></svg></div><figcaption>Excalidraw drawing</figcaption></figure>
      </div>
      <figure class="renderer-specimen actual-excalidraw"><div class="excalidraw-specimen" role="img" aria-label="NOS Excalidraw smoke test drawing"><svg viewBox="0 0 520 360" aria-hidden="true"><path class="drawing-shape" d="M90 180L180 90L270 180L180 270Z"/><text x="180" y="186">Test</text><rect class="drawing-shape" x="300" y="40" width="180" height="108"/><text x="390" y="102">Another Box</text><path class="drawing-line" d="M270 180L360 148"/><rect class="drawing-shape" x="270" y="210" width="145" height="120"/><path class="drawing-line" d="M270 270L415 270"/></svg></div><figcaption>Actual NOS source: <code>Field Notes/FN0003/diagrams/excalidraw-smoke-test.excalidraw</code></figcaption></figure>
    </section>
  {/if}
  {#if saveMessage}<p class="message" role="status">{saveMessage}</p>{/if}
  <section><h2>Semantic colors</h2><div class="swatches">{#each entries(active.colors) as [name, value]}<article class:unresolved={isUnset(value)} class="color-card" style:background={isUnset(value) ? undefined : value} style:color={isUnset(value) ? undefined : readableForeground(value)}>{#if !isUnset(value)}<button class="copy-icon" type="button" aria-label={`Copy ${name} value`} title={`Copy ${name} value`} onclick={() => copyValue(`color:${name}`, value)}>⧉</button>{/if}<div class="color-meta"><strong>{name}</strong>{#if editingColor === `color:${name}`}<input class="color-value" bind:this={colorInput} bind:value={draftColor} aria-label={`Edit ${name}`} onkeydown={(event) => event.key === 'Enter' ? commitColorEdit(name) : event.key === 'Escape' ? cancelColorEdit() : undefined} onblur={() => commitColorEdit(name)} />{:else}<button class="color-value-text" type="button" aria-label={`Edit ${name} value`} onclick={() => beginColorEdit(name, value)}>{isUnset(value) ? 'Unset' : value}</button>{/if}</div></article>{/each}</div></section>
  <section><div class="section-heading"><h2>Custom semantic roles</h2><button class="secondary" type="button" onclick={addCustomRole}>Add role</button></div><div class="swatches">{#each entries(active.customColors) as [name, value]}<article class:unresolved={isUnset(value)} class="color-card" style:background={isUnset(value) ? undefined : value} style:color={isUnset(value) ? undefined : readableForeground(value)}>{#if !isUnset(value)}<button class="copy-icon" type="button" aria-label={`Copy ${name} value`} title={`Copy ${name} value`} onclick={() => copyValue(`custom:${name}`, value)}>⧉</button>{/if}<div class="color-meta"><strong>{name}</strong>{#if editingColor === `custom:${name}`}<input class="color-value" bind:this={colorInput} bind:value={draftColor} aria-label={`Edit ${name}`} onkeydown={(event) => event.key === 'Enter' ? commitColorEdit(name, true) : event.key === 'Escape' ? cancelColorEdit() : undefined} onblur={() => commitColorEdit(name, true)} />{:else}<button class="color-value-text" type="button" aria-label={`Edit ${name} value`} onclick={() => beginColorEdit(name, value, true)}>{isUnset(value) ? 'Unset' : value}</button>{/if}<button class="copy" type="button" onclick={() => removeCustomRole(name)}>Remove role</button></div></article>{/each}</div></section>
  <section><h2>Typography</h2><p class="section-note">Declared roles rendered with their assigned font and type values.</p>{#each entries(family.typography) as [name, style]}<article class="type-row"><div class="type-heading"><code>{name}</code><select class="font-select" value={style.fontFamily} aria-label={`Font family for ${name}`} onchange={(event) => changeFontFamily(name, event.currentTarget.value)}>{#each design.fonts as font}<option value={font.id}>{font.name}</option>{/each}</select></div><p class="type-specimen" style={`font-family:var(--type-${name}-family);font-size:${style.size}px;font-weight:${style.weight};line-height:${style.lineHeight};letter-spacing:${style.letterSpacing ?? 0}px`}>The quick brown fox — 0123456789</p><div class="type-values"><span>{fontName(style.fontFamily)}</span><span>{#if editingValue === `type-weight:${name}`}<input class="inline-editor" bind:this={valueInput} bind:value={draftValue} aria-label={`Edit ${name} weight`} onkeydown={(event) => event.key === 'Enter' ? commitValueEdit('typography', name, 'weight') : event.key === 'Escape' ? cancelValueEdit() : undefined} onblur={() => commitValueEdit('typography', name, 'weight')} />{:else}<button class="value-link" type="button" onclick={() => beginValueEdit(`type-weight:${name}`, style.weight)}>{style.weight}</button>{/if} · {style.style}</span><span>{#if editingValue === `type-size:${name}`}<input class="inline-editor" bind:this={valueInput} bind:value={draftValue} aria-label={`Edit ${name} size`} onkeydown={(event) => event.key === 'Enter' ? commitValueEdit('typography', name, 'size') : event.key === 'Escape' ? cancelValueEdit() : undefined} onblur={() => commitValueEdit('typography', name, 'size')} />{:else}<button class="value-link" type="button" onclick={() => beginValueEdit(`type-size:${name}`, style.size)}>{style.size}px</button>{/if} / {#if editingValue === `type-line:${name}`}<input class="inline-editor" bind:this={valueInput} bind:value={draftValue} aria-label={`Edit ${name} line height`} onkeydown={(event) => event.key === 'Enter' ? commitValueEdit('typography', name, 'line_height') : event.key === 'Escape' ? cancelValueEdit() : undefined} onblur={() => commitValueEdit('typography', name, 'line_height')} />{:else}<button class="value-link" type="button" onclick={() => beginValueEdit(`type-line:${name}`, style.lineHeight)}>{style.lineHeight}</button>{/if}</span><button class="copy-icon inline-copy" type="button" aria-label={`Copy ${name} values`} title={`Copy ${name} values`} onclick={() => copyValue(`type:${name}`, JSON.stringify(style))}>⧉</button></div></article>{/each}</section>
  <section><h2>Spacing</h2><div class="scale-list">{#each entries(family.spacing) as [name, value]}<div><code>{name}</code><span class="space" style:width={`${value}px`}></span><small>{value}px</small><button class="copy" type="button" onclick={() => setFamilyValue('spacing', name, 'value', value)}>Edit</button><button class="copy" type="button" onclick={() => copyValue(`spacing:${name}`, `${value}px`)}>{copied === `spacing:${name}` ? 'Copied' : 'Copy'}</button></div>{/each}</div></section>
  <section><h2>Radii</h2><div class="geometry">{#each entries(family.radii) as [name, value]}<article class="geometry-specimen radius-specimen" style:border-radius={`${valuePreview(`radius:${name}`, value)}px`}><code>{name}</code>{#if editingValue === `radius:${name}`}<input class="inline-editor" bind:this={valueInput} bind:value={draftValue} aria-label={`Edit ${name} radius`} onkeydown={(event) => event.key === 'Enter' ? commitValueEdit('radii', name, 'value') : event.key === 'Escape' ? cancelValueEdit() : undefined} onblur={() => commitValueEdit('radii', name, 'value')} />{:else}<button class="value-link" type="button" onclick={() => beginValueEdit(`radius:${name}`, value)}>{value}px</button>{/if}<button class="copy-icon specimen-copy" type="button" aria-label={`Copy ${name} radius`} title={`Copy ${name} radius`} onclick={() => copyValue(`radius:${name}`, value)}>⧉</button></article>{/each}</div></section>
  <section><h2>Borders</h2><div class="geometry">{#each entries(family.borders) as [name, value]}<article class="geometry-specimen border-specimen" style:border-width={`${valuePreview(`border:${name}`, value)}px`}><code>{name}</code>{#if editingValue === `border:${name}`}<input class="inline-editor" bind:this={valueInput} bind:value={draftValue} aria-label={`Edit ${name} border`} onkeydown={(event) => event.key === 'Enter' ? commitValueEdit('border_widths', name, 'value') : event.key === 'Escape' ? cancelValueEdit() : undefined} onblur={() => commitValueEdit('border_widths', name, 'value')} />{:else}<button class="value-link" type="button" onclick={() => beginValueEdit(`border:${name}`, value)}>{value}px</button>{/if}<button class="copy-icon specimen-copy" type="button" aria-label={`Copy ${name} border`} title={`Copy ${name} border`} onclick={() => copyValue(`border:${name}`, value)}>⧉</button></article>{/each}</div></section>
  <section><h2>Shadows</h2><div class="geometry">{#each entries(family.shadows) as [name, value]}<article class="shadow-specimen"><div class="shadow-object" style:box-shadow={`${editingShadow === name ? draftShadow.x : value.x}px ${editingShadow === name ? draftShadow.y : value.y}px ${editingShadow === name ? draftShadow.blur : value.blur}px ${editingShadow === name ? draftShadow.spread : value.spread}px ${value.color}`}><code>{name}</code><button class="value-link shadow-value" type="button" onclick={() => editingShadow === name ? closeShadowEdit() : beginShadowEdit(name, value)}>{editingShadow === name ? 'Close editor' : `${value.x} ${value.y}px ${value.blur}px ${value.spread}px`}</button></div>{#if editingShadow === name}<div class="shadow-editor"><label>X<input type="number" value={draftShadow.x} oninput={(event) => updateShadowField(name, 'x', event.currentTarget.value)} /></label><label>Y<input type="number" value={draftShadow.y} oninput={(event) => updateShadowField(name, 'y', event.currentTarget.value)} /></label><label>Blur<input type="number" min="0" value={draftShadow.blur} oninput={(event) => updateShadowField(name, 'blur', event.currentTarget.value)} /></label><label>Spread<input type="number" value={draftShadow.spread} oninput={(event) => updateShadowField(name, 'spread', event.currentTarget.value)} /></label></div>{/if}<button class="copy-icon specimen-copy" type="button" aria-label={`Copy ${name} shadow`} title={`Copy ${name} shadow`} onclick={() => copyValue(`shadow:${name}`, JSON.stringify(value))}>⧉</button></article>{/each}</div></section>
  <section><h2>Fonts</h2><p class="section-note">Declared local families and the faces available to typography roles.</p><div class="font-grid">{#each design.fonts as font}<article class="font-specimen surface"><div class="font-heading"><h3>{font.name}</h3><code>{font.id}</code></div><p class="font-sample" style={`font-family: '${font.name}', sans-serif`}>The quick brown fox jumps over the lazy dog.</p><div class="font-faces">{#each font.faces as face}<div style={`font-family: '${font.name}', sans-serif; font-weight:${face.weight}; font-style:${face.style}`}><strong>{face.weight}</strong><span>{humanize(face.style)}</span><p>The quick brown fox 0123456789</p></div>{/each}</div></article>{/each}</div></section>
  {#if design.assets.length > 0}<section><h2>Assets</h2><div class="assets">{#each design.assets as asset}<figure>{#if asset.type === 'image'}<img src={asset.path} alt={asset.name} />{/if}<figcaption>{asset.name}<br /><code>{asset.type}</code></figcaption></figure>{/each}</div></section>{/if}
  <section><h2>Accessibility</h2><div class="contrast">{#each active.contrast as result}<article class:valid={result.valid}><span style:background={token(result.background)} style:color={token(result.foreground)}>Aa</span><div><strong>{result.foreground} on {result.background}</strong><p>{result.ratio}:1 · {result.valid ? 'WCAG AA' : 'Invalid'}</p></div></article>{/each}</div></section>
  <div hidden aria-hidden="true"><div class="shape"></div><nav class="surface"></nav></div>
</main>

<style>
  :global(*){box-sizing:border-box} :global(body){margin:0;background:var(--color-canvas);color:var(--color-content-primary);font-family:var(--type-body-family),sans-serif;font-size:var(--type-body-size);line-height:var(--type-body-line-height)}
  .toolbar{position:sticky;top:0;z-index:2;display:flex;gap:1.25rem;align-items:center;padding:.65rem clamp(1rem,4vw,4rem);background:var(--color-surface-elevated);border-bottom:var(--border-thin,1px) solid var(--color-border-subtle)}.save-controls{display:flex;align-items:center;gap:var(--space-sm,.55rem);margin-left:auto}.save-controls .dirty{color:var(--color-feedback-warning-base)} h1,h2,h3,p{margin-top:0} h1{margin-bottom:0;font-size:var(--type-heading-size)} main{max-width:1200px;margin:auto;padding:2rem clamp(1rem,4vw,4rem)} section{margin-bottom:4rem}.specimen-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:var(--space-md)}
  label,fieldset{display:grid;gap:var(--space-xs,.35rem)}select,input,button{font:inherit;padding:var(--space-sm,.65rem) var(--space-md,.8rem);border:var(--border-thin,1px) solid var(--color-border-default);border-radius:var(--radius-sm);background:var(--color-surface-primary);color:var(--color-content-primary)}select{min-height:2.75rem;appearance:none;-webkit-appearance:none;padding-inline:var(--space-md,.8rem) calc(var(--space-lg,1.5rem) + 2rem);background-image:linear-gradient(45deg,transparent 50%,currentColor 50%),linear-gradient(135deg,currentColor 50%,transparent 50%);background-position:calc(100% - 1.45rem) 50%,calc(100% - 1rem) 50%;background-size:.45rem .45rem,.45rem .45rem;background-repeat:no-repeat;cursor:pointer}button{background:var(--color-accent-primary);color:var(--color-accent-on_primary);font-weight:700;cursor:pointer}.control-row{display:flex;flex-wrap:wrap;gap:var(--space-sm,.55rem);margin-top:var(--space-md,1rem)}.appearance-control{margin:0;padding:0;border:0;background:transparent}.appearance-control legend{margin-bottom:var(--space-xs,.35rem);padding:0}.segments{display:flex;padding:var(--space-xs,.25rem);gap:var(--space-xs,.25rem);border:var(--border-thin,1px) solid var(--color-border-default);border-radius:var(--radius-sm);background:var(--color-surface-primary)}.segments label{position:relative;display:block;cursor:pointer}.segments input{position:absolute;width:1px;height:1px;opacity:0;pointer-events:none}.segments span{display:block;padding:var(--space-sm,.5rem) var(--space-md,.8rem);border-radius:var(--radius-sm);color:var(--color-content-secondary);transition:background-color .15s ease,color .15s ease,box-shadow .15s ease}.segments label:hover span{background:var(--color-surface-secondary);color:var(--color-content-primary)}.segments input:checked+span{background:var(--color-selection-fill);color:var(--color-selection-content);box-shadow:inset 0 0 0 var(--border-thin,1px) var(--color-selection-border)}.segments input:focus-visible+span,select:focus-visible,button:focus-visible,.checkbox-row input:focus-visible{outline:var(--border-strong,2px) solid var(--color-focus-ring);outline-offset:2px}.checkbox-row{display:flex;align-items:center;gap:var(--space-sm,.55rem);margin-top:var(--space-xs,.35rem);cursor:pointer}.checkbox-row input{width:1rem;height:1rem;margin:0;padding:0;accent-color:var(--color-accent-primary);cursor:pointer}button:hover{filter:brightness(1.08)}select:hover{border-color:var(--color-border-strong)}button:disabled,select:disabled,input:disabled+span{cursor:not-allowed;opacity:.6}.secondary{background:var(--color-accent-secondary);color:var(--color-accent-on_secondary)}.surface{padding:var(--space-md);background:var(--color-surface-elevated);border:var(--border-thin,1px) solid var(--color-border-subtle);border-radius:var(--radius-md);box-shadow:var(--shadow-low)}.eyebrow{text-transform:uppercase;letter-spacing:.12em;font-size:.72rem;color:var(--color-content-muted);margin-bottom:.25rem}.message{padding:var(--space-md);background:var(--color-feedback-info-container);color:var(--color-feedback-info-on_container)}
  .swatches{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,220px),1fr));gap:1rem}.swatches article{min-width:0;max-width:100%;overflow:hidden}.color-card{position:relative;display:flex;min-height:6.25rem;padding:var(--space-sm,.65rem);border:var(--border-thin,1px) solid color-mix(in srgb,currentColor 35%,transparent);border-radius:var(--radius-sm);overflow:hidden}.color-meta{position:relative;z-index:1;display:flex;flex-direction:column;justify-content:flex-end;align-items:flex-start;align-self:stretch;width:100%;min-width:0;padding:.25rem .15rem;background:color-mix(in srgb,currentColor 12%,transparent);border-radius:var(--radius-sm);text-shadow:0 1px 2px color-mix(in srgb,currentColor 25%,transparent)}.color-card strong{min-width:0;max-width:calc(100% - 2.2rem);overflow-wrap:break-word;word-break:normal;line-height:1.2}.color-value{min-width:0;width:10rem;max-width:100%;padding:.2rem .35rem;border-color:color-mix(in srgb,currentColor 45%,transparent);background:color-mix(in srgb,currentColor 12%,transparent);color:inherit;font-family:var(--type-code-family),monospace;font-size:var(--type-code-size)}.color-value-text{max-width:100%;padding:0;border:0;background:transparent;color:inherit;font-family:var(--type-code-family),monospace;font-size:var(--type-code-size);text-align:left;cursor:text;opacity:.9}.color-value-text:hover,.color-value-text:focus-visible{opacity:1;text-decoration:underline;text-underline-offset:.2em}code,small{display:block;overflow-wrap:break-word;word-break:normal;color:var(--color-content-muted)}.copy{justify-self:start;padding:var(--space-xs,.3rem) var(--space-sm,.55rem);font-size:.78rem;background:var(--color-surface-secondary);color:var(--color-content-primary)}.copy-icon{position:absolute;top:var(--space-xs,.35rem);right:var(--space-xs,.35rem);z-index:2;display:grid;place-items:center;width:1.8rem;height:1.8rem;padding:0;border:var(--border-thin,1px) solid color-mix(in srgb,currentColor 45%,transparent);border-radius:var(--radius-sm);font-size:1rem;line-height:1;background:color-mix(in srgb,currentColor 16%,transparent);color:inherit}.type-row{padding:1rem 0;border-bottom:1px solid var(--color-border-subtle)}.type-row p{margin:.5rem 0}.scale-list{display:grid;gap:.75rem}.scale-list div{display:grid;grid-template-columns:8rem 1fr 4rem auto;align-items:center;gap:var(--space-sm,.55rem)}.space{display:block;height:1.5rem;background:var(--color-accent-primary)}.geometry,.assets{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,180px),1fr));gap:1.5rem}.shape{min-height:140px;display:grid;place-content:center;text-align:center;background:var(--color-surface-elevated);border-style:solid;border-color:var(--color-border-default)}figure{margin:0;padding:1rem;border:1px solid var(--color-border-subtle)}figure img{width:100%;height:120px;object-fit:contain}.contrast{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,320px),1fr));gap:1.5rem}.contrast article{display:grid;grid-template-columns:72px minmax(0,1fr);gap:1rem;align-items:center;min-width:0;overflow:hidden;padding:1rem;border:2px solid var(--color-feedback-danger-base)}.contrast article>div{min-width:0}.contrast strong{overflow-wrap:break-word;word-break:normal}.contrast article.valid{border-color:var(--color-feedback-success-base)}.contrast span{display:grid;place-items:center;width:72px;height:72px;font-size:1.5rem;font-weight:700}.contrast p{margin:0;overflow-wrap:break-word;word-break:normal}@media(max-width:720px){.toolbar{position:static;display:grid}.scale-list div{grid-template-columns:5rem 1fr 3rem auto}nav.surface{flex-wrap:wrap}.contrast{grid-template-columns:minmax(0,1fr)}}
  .editorial-specimen{max-width:var(--content-width-reading,42rem);font-family:var(--type-body-family),serif}.editorial-specimen h3{font-family:var(--type-heading-family),sans-serif;font-size:var(--type-heading-size);line-height:var(--type-heading-line-height)}.editorial-lead{font-size:1.15em;color:var(--color-content-secondary)}.editorial-specimen a{color:var(--color-accent-primary);text-decoration:underline;text-underline-offset:.2em}.editorial-specimen blockquote{margin:1.5rem 0;padding:.75rem 1rem;border-inline-start:var(--border-strong,2px) solid var(--color-accent-primary);color:var(--color-content-secondary)}.editorial-figure{margin-block:1.5rem}.media-placeholder,.diagram-placeholder,.drawing-placeholder{display:grid;min-height:8rem;place-items:center;border:var(--border-thin,1px) solid var(--color-border-default);background:var(--color-surface-secondary);color:var(--color-content-muted)}.editorial-figure figcaption,.renderer-specimen figcaption{margin-top:.65rem;color:var(--color-content-muted);font-size:.85em}.syntax-specimen{overflow:auto;margin:1.5rem 0;padding:var(--space-md);border:var(--border-thin,1px) solid var(--color-renderer-syntax-border,var(--color-border-subtle));border-radius:var(--radius-sm);background:var(--color-renderer-syntax-surface,var(--color-surface-secondary));color:var(--color-renderer-syntax-text,var(--color-content-primary));font-family:var(--type-code-family),monospace}.syntax-keyword{color:var(--color-renderer-syntax-keyword)}.syntax-string{color:var(--color-renderer-syntax-literal)}.syntax-name{color:var(--color-renderer-syntax-identifier)}.syntax-type{color:var(--color-renderer-syntax-type)}.syntax-punctuation{color:var(--color-renderer-syntax-punctuation)}.content-specimens .renderer-grid{display:none}.renderer-examples{margin-top:var(--space-lg)}.renderer-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:var(--space-md)}.renderer-specimen{margin:0}.mermaid-specimen,.excalidraw-specimen{display:grid;min-height:8rem;place-items:center;overflow:auto;padding:var(--space-sm);border:var(--border-thin,1px) solid var(--color-renderer-diagram,var(--color-border-default));background:var(--color-renderer-diagram,var(--color-surface-secondary));color:var(--color-renderer-diagram-label,var(--color-content-primary))}.excalidraw-specimen{border-color:var(--color-renderer-drawing,var(--color-border-default));background:var(--color-renderer-drawing,var(--color-surface-secondary));color:var(--color-renderer-drawing-label,var(--color-content-primary))}.mermaid-specimen :global(svg),.excalidraw-specimen :global(svg){width:100%;max-width:24rem;height:auto}.mermaid-specimen :global(text),.excalidraw-specimen :global(text){fill:currentColor;font:600 12px var(--type-code-family),monospace;text-anchor:middle}.mermaid-specimen :global(.diagram-node){fill:var(--color-renderer-diagram-node,var(--color-surface-primary));stroke:currentColor}.mermaid-specimen :global(.diagram-edge){stroke:currentColor;stroke-width:2}.drawing-shape{fill:var(--color-renderer-drawing-node,var(--color-surface-primary));stroke:currentColor;stroke-width:2}.drawing-line{stroke:currentColor;stroke-width:2}.renderer-source{margin-top:var(--space-sm);padding:var(--space-sm);overflow:auto;background:var(--color-renderer-syntax-surface,var(--color-surface-secondary));color:var(--color-renderer-syntax-text,var(--color-content-primary));font:.8rem var(--type-code-family),monospace}
  /* Compact specimen inset and toolbar refinements. */
  .color-card { padding: var(--space-sm, .65rem); }
  .color-meta { padding: 0; background: transparent; text-shadow: none; }
  .color-meta strong, .color-value-text, .color-value { text-shadow: 0 1px 2px color-mix(in srgb, currentColor 45%, transparent); }
  .copy-icon { border-color: transparent; }
  .copy-icon { font-size: 0; }
  .copy-icon::before { content: ''; width: 1rem; height: 1rem; background: currentColor; -webkit-mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Crect x='8' y='8' width='11' height='11' rx='1.5' fill='none' stroke='black' stroke-width='1.8'/%3E%3Cpath d='M16 8V5.5A1.5 1.5 0 0 0 14.5 4h-9A1.5 1.5 0 0 0 4 5.5v9A1.5 1.5 0 0 0 5.5 16H8' fill='none' stroke='black' stroke-width='1.8' stroke-linecap='round'/%3E%3C/svg%3E") center/contain no-repeat; mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Crect x='8' y='8' width='11' height='11' rx='1.5' fill='none' stroke='black' stroke-width='1.8'/%3E%3Cpath d='M16 8V5.5A1.5 1.5 0 0 0 14.5 4h-9A1.5 1.5 0 0 0 4 5.5v9A1.5 1.5 0 0 0 5.5 16H8' fill='none' stroke='black' stroke-width='1.8' stroke-linecap='round'/%3E%3C/svg%3E") center/contain no-repeat; }
  .copy-icon:hover, .copy-icon:focus-visible { border-color: color-mix(in srgb, currentColor 45%, transparent); }
  .copy-icon::before { position: absolute; top: 50%; left: 50%; display: block; margin: 0; transform: translate(-50%, -50%); }
  .copy-icon { display: flex; align-items: center; justify-content: center; line-height: 0; }
  .copy-icon::before { position: static; flex: 0 0 1rem; transform: none; }
  .copy-icon::before { position: relative; left: 1px; top: 1px; }
  .toolbar { gap: var(--space-md, 1rem); padding-block: var(--space-xs, .35rem); min-height: 4.25rem; }
  .appearance-control { display: flex; align-items: center; gap: var(--space-xs, .35rem); }
  .appearance-control legend { margin: 0; }
  .section-heading { display:flex; align-items:center; justify-content:space-between; gap:var(--space-md,1rem); }
  .save-controls { margin-left: auto; }
  .management-row { display: flex; flex-wrap: wrap; align-items: center; gap: var(--space-sm, .55rem); }
  @media(max-width: 720px) {
    .toolbar { gap: var(--space-sm, .55rem); }
    .save-controls { margin-left: 0; }
  }
  .toolbar { flex-wrap: wrap; gap: var(--space-md, 1rem); padding-block: var(--space-sm, .55rem); min-height: 0; }
  .identity { flex: 0 0 auto; }
  .toolbar-controls { display: flex; align-items: center; flex-wrap: wrap; gap: var(--space-md, 1rem); margin-left: auto; }
  .save-controls { margin-left: 0; min-height: 2.75rem; }
  .theme-menu { position: relative; }
  .theme-menu summary { display: grid; place-items: center; width: 2.75rem; height: 2.75rem; padding: 0; border: var(--border-thin, 1px) solid var(--color-border-default); border-radius: var(--radius-sm); color: var(--color-content-secondary); cursor: pointer; list-style: none; letter-spacing: .12em; }
  .theme-menu summary::-webkit-details-marker { display: none; }
  .theme-menu summary:hover { color: var(--color-content-primary); background: var(--color-surface-secondary); }
  .theme-menu summary:focus-visible { outline: var(--border-strong, 2px) solid var(--color-focus-ring); outline-offset: 2px; }
  .theme-menu-panel { position: absolute; top: calc(100% + var(--space-sm, .55rem)); right: 0; z-index: 4; display: grid; gap: var(--space-sm, .55rem); width: min(21rem, calc(100vw - 2rem)); padding: var(--space-md, 1rem); background: var(--color-surface-elevated); border: var(--border-thin, 1px) solid var(--color-border-default); border-radius: var(--radius-md); box-shadow: var(--shadow-medium, var(--shadow-low)); }
  .menu-heading, .appearance-row { display: flex; align-items: center; justify-content: space-between; gap: var(--space-sm, .55rem); }
  .menu-heading span, .supported { color: var(--color-content-muted); font-size: .82em; }
  .menu-label { margin: var(--space-sm, .55rem) 0 0; color: var(--color-content-muted); font-size: .78em; font-weight: 700; letter-spacing: .08em; }
  .appearance-list { display: grid; gap: var(--space-xs, .35rem); padding-top: var(--space-sm, .55rem); border-top: var(--border-thin, 1px) solid var(--color-border-subtle); }
  .appearance-row { min-height: 2.25rem; }
  .text-button { padding: var(--space-xs, .3rem) var(--space-sm, .55rem); font-size: .8em; }
  .danger { background: transparent; color: var(--color-feedback-danger-base); border-color: color-mix(in srgb, var(--color-feedback-danger-base) 45%, transparent); }
  .danger:hover { background: color-mix(in srgb, var(--color-feedback-danger-base) 10%, transparent); }
  main { padding-top: var(--space-md, 1rem); }
  main section { margin-bottom: var(--space-xl, 3rem); }
  .context { display: flex; flex-wrap: wrap; justify-content: space-between; gap: var(--space-xs, .35rem) var(--space-lg, 1.5rem); padding: 0 0 var(--space-md, 1rem); color: var(--color-content-muted); border-bottom: var(--border-thin, 1px) solid var(--color-border-subtle); font-size: .9em; }
  .context strong { color: var(--color-content-secondary); font-weight: 600; }
  .mobile-shell-specimen { margin: var(--space-xl, 2rem) 0; }
  .section-heading { display: flex; align-items: baseline; justify-content: space-between; gap: 1rem; }
  .specimen-badge { color: var(--color-content-secondary); border: var(--border-thin, 1px) solid var(--color-border-default); border-radius: var(--radius-sm, .35rem); padding: .35rem .55rem; font-size: .8em; }
  .mobile-shell-frame { min-height: 22rem; display: flex; flex-direction: column; overflow: hidden; color: var(--color-content-primary); background: var(--color-surface-primary); border: var(--border-thin, 1px) solid var(--color-border-default); border-radius: var(--radius-md, .75rem); box-shadow: var(--shadow-low, none); padding: env(safe-area-inset-top, 0px) env(safe-area-inset-right, 0px) env(safe-area-inset-bottom, 0px) env(safe-area-inset-left, 0px); }
  .mobile-shell-header, .mobile-shell-footer { display: flex; align-items: center; justify-content: space-between; gap: .75rem; padding: .8rem 1rem; background: var(--color-surface-elevated); border-bottom: var(--border-thin, 1px) solid var(--color-border-subtle); }
  .mobile-shell-footer { border-top: var(--border-thin, 1px) solid var(--color-border-subtle); border-bottom: 0; color: var(--color-content-muted); }
  .mobile-shell-status { color: var(--color-accent-primary); }
  .mobile-shell-body { position: relative; display: flex; flex: 1; justify-content: space-between; gap: 1rem; padding: 1.25rem; }
  .mobile-shell-body h3 { margin: .25rem 0; color: var(--color-content-primary); }
  .mobile-shell-body p { max-width: 36rem; color: var(--color-content-secondary); }
  .mobile-shell-overlay { align-self: start; color: var(--color-content-primary); background: var(--color-surface-elevated); border: var(--border-thin, 1px) solid var(--color-border-subtle); border-radius: var(--radius-sm, .35rem); padding: .6rem .75rem; }
  .mobile-shell-footer button { color: var(--color-accent-on_primary); background: var(--color-accent-primary); border: 0; border-radius: var(--radius-sm, .35rem); padding: .45rem .7rem; }
  .mobile-shell-footer button:focus-visible { outline: var(--border-strong, 2px) solid var(--color-focus-ring); outline-offset: 2px; }
  @media (max-width: 42rem) { .mobile-shell-body { flex-direction: column; } .mobile-shell-overlay { align-self: stretch; } }
  main > section:first-of-type h2 { margin-bottom: var(--space-xs, .35rem); }
  main > section:first-of-type > p { color: var(--color-content-muted); }
  .color-card { min-height: 7rem; padding: var(--space-md, 1rem); border-color: color-mix(in srgb, currentColor 38%, transparent); }
  .color-meta { justify-content: flex-end; gap: var(--space-xs, .35rem); padding: 0; background: transparent; border-radius: 0; }
  .color-meta strong { max-width: calc(100% - 2.5rem); }
  .color-value-text { padding: 0; opacity: .78; }
  .color-value-text:hover, .color-value-text:focus-visible { opacity: 1; }
  .copy-icon { top: var(--space-md, 1rem); right: var(--space-md, 1rem); background: transparent; }
  .copy-icon:hover, .copy-icon:focus-visible { background: color-mix(in srgb, currentColor 12%, transparent); }
  @media(max-width: 900px) {
    .toolbar-controls { margin-left: 0; }
    .save-controls { margin-left: auto; }
  }
  /* The toolbar is one control grammar: shared labels, heights, and rhythm. */
  .toolbar { align-items: center; }
  .identity { display: grid; align-content: center; gap: 0; min-height: 2.75rem; }
  .identity .eyebrow { margin-bottom: .1rem; }
  .bench-version { margin-left: var(--space-xs, .35rem); color: var(--color-content-muted); font-size: .55em; font-weight: 500; letter-spacing: normal; vertical-align: middle; }
  .toolbar-controls { align-items: end; gap: var(--space-lg, 1.5rem); }
  .theme-control { display: grid; gap: var(--space-xs, .35rem); }
  .toolbar-label, .appearance-control legend { color: var(--color-content-secondary); font-size: .78em; font-weight: 700; line-height: 1; }
  .theme-control-shell { display: flex; align-items: stretch; height: 2.75rem; }
  .theme-control-shell { position: relative; border-radius: var(--radius-sm, 0); }
  .theme-control-shell:focus-within { border-radius: var(--radius-sm, 0); }
  .theme-control-shell:has(select:focus) { border-color: transparent; }
  .theme-control-shell select:focus, .theme-control-shell summary:focus-visible { outline: none; box-shadow: none; }
  .toolbar select:focus-visible, .toolbar button:focus-visible { outline: none; }
  .theme-control-shell select:focus { border-color: var(--color-border-strong); color: var(--color-content-primary); }
  .theme-menu summary:focus-visible { border-color: var(--color-border-strong); color: var(--color-content-primary); background: var(--color-surface-secondary); }
  .appearance-control .segments input:focus-visible + span { outline: none; box-shadow: inset 0 0 0 var(--border-thin, 1px) var(--color-border-strong); }
  .toolbar button:focus-visible { border-color: var(--color-border-strong); color: var(--color-content-primary); }
  .theme-select { display: block; gap: 0; }
  .theme-select select { display: block; width: 18rem; height: 2.75rem; min-height: 2.75rem; border-radius: var(--radius-sm) 0 0 var(--radius-sm); }
  .theme-menu { height: 2.75rem; }
  .theme-menu summary { width: 2.75rem; height: 2.75rem; border-left: 0; border-radius: 0 var(--radius-sm) var(--radius-sm) 0; background: var(--color-surface-primary); }
  .theme-menu[open] summary { background: var(--color-surface-secondary); color: var(--color-content-primary); }
  .appearance-control { display: grid; align-content: start; gap: var(--space-xs, .35rem); }
  .appearance-control legend { margin: 0; padding: 0; }
  .segments { height: 2.75rem; align-items: center; padding: var(--space-xs, .25rem); }
  .segments span { min-height: 2.15rem; display: grid; place-items: center; padding-block: var(--space-xs, .35rem); }
  .segments span svg { width: 1rem; height: 1rem; fill: none; stroke: currentColor; stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; }
  .save-controls { align-self: end; min-height: 2.75rem; }
  .save-controls .dirty { font-size: .9em; }
  .theme-menu-panel { top: calc(100% + var(--space-xs, .35rem)); right: 0; gap: var(--space-md, 1rem); padding: var(--space-md, 1rem); background: var(--color-surface-elevated); color: var(--color-content-primary); border-color: var(--color-border-default); box-shadow: var(--shadow-low); }
  .theme-menu-panel .management-row { gap: var(--space-sm, .55rem); }
  .theme-menu-panel .management-row button { min-height: 2.35rem; }
  .menu-heading { padding-bottom: var(--space-xs, .35rem); }
  .menu-heading strong { font-size: 1rem; }
  .menu-heading span { padding: .2rem .45rem; border-radius: 999px; background: var(--color-selection-fill); color: var(--color-selection-content); font-size: .75em; font-weight: 700; }
  .appearance-list { gap: var(--space-xs, .35rem); margin-top: var(--space-xs, .35rem); padding-top: var(--space-md, 1rem); }
  .menu-label { margin: 0 0 var(--space-xs, .35rem); }
  .appearance-row { min-height: 2.35rem; padding-block: var(--space-xs, .35rem); }
  .appearance-row > span:first-child { min-width: 4.5rem; }
  .appearance-row .supported { flex: 1; }
  .text-button { min-width: 4.5rem; }
  .danger { background: var(--color-feedback-danger-container); color: var(--color-feedback-danger-content); border-color: color-mix(in srgb, var(--color-feedback-danger-base) 48%, var(--color-border-default)); font-weight: 600; }
  .danger:hover, .danger:focus-visible { background: var(--color-feedback-danger-base); color: var(--color-feedback-danger-on_container); border-color: var(--color-feedback-danger-base); }
  .danger:disabled { background: var(--color-surface-secondary); color: var(--color-content-muted); border-color: var(--color-border-subtle); }
  @media(max-width: 720px) {
    .toolbar { align-items: start; }
    .toolbar-controls { width: 100%; align-items: end; gap: var(--space-md, 1rem); }
    .theme-control-shell, .theme-select select, .segments, .theme-menu, .theme-menu summary { height: 2.75rem; }
    .theme-select select { width: min(18rem, calc(100vw - 8rem)); }
    .save-controls { margin-left: 0; }
  }
  /* Structural layers: content stays below the persistent chrome; only intentional overlays rise above it. */
  .toolbar { z-index: 10; isolation: isolate; background-color: var(--color-surface-elevated); }
  main { position: relative; z-index: 0; }
  .theme-menu-panel { z-index: 20; }
  .copy-icon { z-index: 1; }
  .section-note { color: var(--color-content-muted); }
  .geometry-specimen, .shadow-specimen { position: relative; min-width: 0; min-height: 8rem; display: grid; place-items: center; gap: var(--space-xs, .35rem); padding: var(--space-md, 1rem); background: var(--color-surface-elevated); border: var(--border-thin, 1px) solid var(--color-border-default); text-align: center; }
  .radius-specimen { background: var(--color-surface-secondary); }
  .border-specimen { border-color: var(--color-content-secondary); }
  .geometry-specimen code, .shadow-specimen code { color: var(--color-content-secondary); }
  .value-link { display: inline; width: auto; min-width: 0; padding: 0; border: 0; background: transparent; color: var(--color-content-primary); font-family: var(--type-code-family), monospace; font-size: var(--type-code-size); font-weight: 600; cursor: text; }
  .value-link:hover, .value-link:focus-visible { text-decoration: underline; text-underline-offset: .2em; }
  .inline-editor { width: 6rem; min-width: 0; padding: var(--space-xs, .3rem) var(--space-sm, .55rem); background: var(--color-surface-primary); color: var(--color-content-primary); font-family: var(--type-code-family), monospace; }
  .specimen-copy { top: var(--space-sm, .55rem); right: var(--space-sm, .55rem); }
  .shadow-specimen { min-height: 13rem; align-content: center; }
  .shadow-object { display: grid; place-items: center; gap: var(--space-xs, .35rem); width: min(100%, 12rem); min-height: 7rem; padding: var(--space-md, 1rem); background: var(--color-surface-primary); border: var(--border-thin, 1px) solid var(--color-border-subtle); border-radius: var(--radius-md); }
  .shadow-value { cursor: text; }
  .shadow-editor { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: var(--space-xs, .35rem); width: min(100%, 18rem); padding-top: var(--space-sm, .55rem); border-top: var(--border-thin, 1px) solid var(--color-border-subtle); text-align: left; }
  .shadow-editor label { display: grid; gap: .2rem; color: var(--color-content-muted); font-size: .75em; }
  .shadow-editor input { width: 100%; min-width: 0; padding: var(--space-xs, .3rem); }
  .type-row { padding: var(--space-md, 1rem) 0; }
  .type-heading, .font-heading, .type-values { display: flex; align-items: center; gap: var(--space-sm, .55rem); }
  .type-heading { justify-content: space-between; }
  .type-specimen { margin: var(--space-md, 1rem) 0; color: var(--color-content-primary); }
  .type-values { position: relative; flex-wrap: wrap; color: var(--color-content-muted); font-size: .85em; }
  .type-values > span { display: inline-flex; align-items: center; gap: .2rem; }
  .type-values .value-link { color: var(--color-content-secondary); font-size: 1em; }
  .font-select { min-height: 2.35rem; padding-block: var(--space-xs, .3rem); }
  .inline-copy { position: static; margin-left: auto; }
  .font-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 23rem), 1fr)); gap: var(--space-md, 1rem); }
  .font-specimen { min-width: 0; }
  .font-heading { justify-content: space-between; }
  .font-heading h3 { margin-bottom: 0; }
  .font-sample { margin: var(--space-md, 1rem) 0; font-size: 1.35rem; line-height: 1.35; }
  .font-faces { display: grid; gap: var(--space-sm, .55rem); }
  .font-faces > div { padding-top: var(--space-sm, .55rem); border-top: var(--border-thin, 1px) solid var(--color-border-subtle); }
  .font-faces strong { margin-right: var(--space-xs, .35rem); color: var(--color-content-primary); }
  .font-faces span, .font-faces p { color: var(--color-content-muted); font-size: .85em; }
  .font-faces p { margin: .35rem 0 0; font-size: 1.05rem; color: var(--color-content-primary); }
  /* Keep the projected geometry and spacing grammar continuous through the lower page. */
  main > section { margin-bottom: var(--space-lg, var(--space-md, 1rem)); }
  main > section > h2 { margin-bottom: var(--space-sm, .55rem); }
  .swatches, .geometry, .assets, .contrast, .font-grid, .specimen-grid { gap: var(--space-md, var(--space-sm, .55rem)); }
  .geometry-specimen, .shadow-specimen, .font-specimen, .contrast article, figure { border-radius: var(--radius-md, var(--radius-sm, 0)); }
  .geometry-specimen, .shadow-specimen, .font-specimen, .contrast article, figure { border-width: var(--border-thin, 1px); }
  .geometry-specimen, .shadow-specimen, .font-specimen, .contrast article, figure { padding: var(--space-md, var(--space-sm, .55rem)); }
  .type-row { padding-block: var(--space-md, var(--space-sm, .55rem)); border-bottom-width: var(--border-thin, 1px); }
  .type-row + .type-row { margin-top: 0; }
  .contrast article { border-style: solid; }
  .contrast span { width: 4.5rem; height: 4.5rem; border-radius: var(--radius-sm, 0); }
  figure { margin: 0; }
  .shadow-editor { gap: var(--space-xs, .35rem); padding-top: var(--space-sm, .55rem); }
  @media(max-width: 720px) { .shadow-editor { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
  :global(html), :global(body) { background-color: var(--color-canvas); }
  .color-card { transition: transform .15s ease, box-shadow .15s ease; }
  .color-card:hover, .color-card:focus-within { transform: translateY(-2px); box-shadow: var(--shadow-low); }
  .color-card.unresolved { background: repeating-linear-gradient(-45deg, color-mix(in srgb, var(--color-surface-primary) 92%, transparent) 0 8px, color-mix(in srgb, var(--color-border-subtle) 72%, transparent) 8px 10px); border-style: dashed; border-color: var(--color-border-strong); color: var(--color-content-secondary); }
  .color-card.unresolved .color-meta { background: transparent; text-shadow: none; }
  .copy-icon:hover, .copy-icon:focus-visible { filter: brightness(1.15); transform: scale(1.05); }
  .color-value-text:hover, .color-value-text:focus-visible { color: var(--color-accent-primary); }
  /* Editorial-only themes must not expose renderer specimens embedded in the
     shared content sample. Field Notes renders the gated replacements above. */
  .content-specimens .syntax-specimen, .content-specimens .renderer-grid { display: none; }
  .renderer-examples > .renderer-grid > figure:nth-child(2) { display: none; }
  .appearance-control:has(.segments label:only-child) { display: none; }
</style>
