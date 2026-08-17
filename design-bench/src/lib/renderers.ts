import type { Highlighter } from 'shiki';

const source = `defmodule Bench.FieldNotes do
  @moduledoc "A renderer-facing Design Bench specimen."

  @type signal :: :visible | {:error, String.t()}

  def publish(%{title: title} = note) when is_binary(title) do
    %{note: note, signal: :visible}
  end
end`;

let highlighter: Promise<Highlighter> | undefined;
let mermaidModule: Promise<typeof import('mermaid')> | undefined;

const color = (appearance: any, role: string, fallback: string) =>
  appearance?.customColors?.[`renderer.${role}`] ?? fallback;

const syntaxTheme = (appearance: any) => ({
  name: 'design-bench-semantic',
  type: 'dark',
  colors: {
    'editor.background': color(appearance, 'syntax.surface', '#121212'),
    'editor.foreground': color(appearance, 'syntax.text', '#DBD7CA')
  },
  settings: [
    { settings: { foreground: color(appearance, 'syntax.text', '#DBD7CA') } },
    { scope: ['comment'], settings: { foreground: color(appearance, 'syntax.punctuation', '#666666') } },
    { scope: ['keyword', 'storage'], settings: { foreground: color(appearance, 'syntax.keyword', '#4D9375') } },
    { scope: ['entity.name.type', 'support.type'], settings: { foreground: color(appearance, 'syntax.type', '#5DA994') } },
    { scope: ['entity.name', 'variable'], settings: { foreground: color(appearance, 'syntax.identifier', '#80A665') } },
    { scope: ['string', 'constant.numeric', 'constant.language'], settings: { foreground: color(appearance, 'syntax.literal', '#C99076') } },
    { scope: ['punctuation', 'keyword.operator'], settings: { foreground: color(appearance, 'syntax.punctuation', '#666666') } }
  ]
});

export const renderSyntax = async (appearance: any) => {
  const { createHighlighter } = await import('shiki');
  highlighter ??= createHighlighter({ langs: ['elixir'], themes: [syntaxTheme(appearance) as any] });
  const instance = await highlighter;
  instance.loadTheme(syntaxTheme(appearance) as any);
  return instance.codeToHtml(source, { lang: 'elixir', theme: 'design-bench-semantic' });
};

export const renderDiagram = async (appearance: any, id: string) => {
  mermaidModule ??= import('mermaid');
  const { default: mermaid } = await mermaidModule;
  const light = appearance?.customColors?.['renderer.diagram'] ?? '#F4F0E7';
  const text = appearance?.colors?.['content.primary'] ?? '#1A1B17';
  const border = appearance?.customColors?.['renderer.diagram.border'] ?? appearance?.colors?.['border.default'] ?? '#AAA59A';
  mermaid.initialize({
    startOnLoad: false,
    securityLevel: 'strict',
    theme: 'base',
    themeVariables: {
      background: appearance?.colors?.canvas ?? light,
      primaryColor: light,
      primaryTextColor: text,
      primaryBorderColor: border,
      lineColor: appearance?.colors?.['content.secondary'] ?? text,
      textColor: text,
      clusterBkg: appearance?.colors?.['surface.secondary'] ?? light,
      clusterBorder: appearance?.colors?.['border.subtle'] ?? border,
      edgeLabelBackground: light,
      fontFamily: 'var(--type-body-family), sans-serif'
    }
  });
  const result = await mermaid.render(id, 'flowchart LR\n  Intent --> Architecture\n  Architecture --> Implementation\n  Implementation --> System');
  return result.svg;
};
