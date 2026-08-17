// Derived from design/system.yml. Project-owned; do not edit by hand.
export type AppearanceName = 'light' | 'dark';
export interface DesignMetadata { defaultTheme: string; defaultAppearance: 'system' | AppearanceName; themes: DesignTheme[]; fonts: DesignFont[]; assets: DesignAsset[] }
export interface DesignTheme { id: string; typography: Record<string, TextStyle>; spacing: Record<string, number>; radii: Record<string, number>; borders: Record<string, number>; shadows: Record<string, Shadow>; appearances: Partial<Record<AppearanceName, DesignAppearance>> }
export interface DesignAppearance { colors: Record<string, string | null>; customColors: Record<string, string | null>; contrast: ContrastResult[] }
export interface TextStyle { fontFamily: string; size: number; weight: number; lineHeight: number; letterSpacing: number; style: string }
export interface Shadow { color: string; x: number; y: number; blur: number; spread: number }
export interface ContrastResult { foreground: string; background: string; ratio: number; valid: boolean }
export interface DesignFont { id: string; name: string; faces: Array<{ asset: string; weight: number; style: string }> }
export interface DesignAsset { name: string; path: string; type: string }

export const design: DesignMetadata = {
  "themes": [
    {
      "id": "sibyl",
      "appearances": {
        "light": {
          "colors": {
            "feedback.danger.content": "#102126",
            "disabled.content": "#55727A",
            "feedback.success.container": "#D8F3F5",
            "selection.fill": "#146A9C",
            "content.primary": "#102126",
            "feedback.success.content": "#102126",
            "focus.ring": "#146A9C",
            "feedback.warning.container": "#D8F3F5",
            "feedback.success.on_container": "#102126",
            "surface.elevated": "#E8F5F7",
            "feedback.warning.on_container": "#102126",
            "feedback.info.on_container": "#102126",
            "feedback.danger.base": "#B83A52",
            "feedback.danger.on_container": "#102126",
            "surface.primary": "#FFFFFF",
            "feedback.warning.content": "#102126",
            "disabled.surface": "#E8F5F7",
            "accent.on_primary": "#FFFFFF",
            "selection.border": "#146A9C",
            "accent.primary": "#146A9C",
            "accent.on_secondary": "#FFFFFF",
            "canvas": "#F4FBFC",
            "border.strong": "#102126",
            "content.secondary": "#31515A",
            "feedback.info.content": "#146A9C",
            "border.subtle": "#A9C9CE",
            "feedback.info.container": "#D8F3F5",
            "selection.content": "#FFFFFF",
            "feedback.warning.base": "#B97811",
            "feedback.info.base": "#146A9C",
            "content.muted": "#55727A",
            "accent.secondary": "#146A9C",
            "feedback.success.base": "#2AAE83",
            "surface.secondary": "#F4FBFC",
            "border.default": "#31515A",
            "feedback.danger.container": "#D8F3F5"
          },
          "contrast": [
            {
              "valid": true,
              "background": "canvas",
              "foreground": "content.primary",
              "ratio": 15.82
            },
            {
              "valid": true,
              "background": "surface.primary",
              "foreground": "content.primary",
              "ratio": 16.57
            },
            {
              "valid": true,
              "background": "canvas",
              "foreground": "content.secondary",
              "ratio": 8.17
            },
            {
              "valid": true,
              "background": "canvas",
              "foreground": "content.muted",
              "ratio": 4.92
            },
            {
              "valid": true,
              "background": "accent.primary",
              "foreground": "accent.on_primary",
              "ratio": 5.88
            },
            {
              "valid": true,
              "background": "accent.secondary",
              "foreground": "accent.on_secondary",
              "ratio": 5.88
            },
            {
              "valid": true,
              "background": "feedback.info.container",
              "foreground": "feedback.info.on_container",
              "ratio": 14.25
            },
            {
              "valid": true,
              "background": "feedback.success.container",
              "foreground": "feedback.success.on_container",
              "ratio": 14.25
            },
            {
              "valid": true,
              "background": "feedback.warning.container",
              "foreground": "feedback.warning.on_container",
              "ratio": 14.25
            },
            {
              "valid": true,
              "background": "feedback.danger.container",
              "foreground": "feedback.danger.on_container",
              "ratio": 14.25
            },
            {
              "valid": true,
              "background": "selection.fill",
              "foreground": "selection.content",
              "ratio": 5.88
            }
          ],
          "customColors": {
            "identity.oracle_blue": "#146A9C"
          }
        },
        "dark": {
          "colors": {
            "feedback.danger.content": "#F1FEFF",
            "disabled.content": "#7DA8AE",
            "feedback.success.container": "#123B45",
            "selection.fill": "#1FB3FF",
            "content.primary": "#F1FEFF",
            "feedback.success.content": "#F1FEFF",
            "focus.ring": "#1FB3FF",
            "feedback.warning.container": "#123B45",
            "feedback.success.on_container": "#F1FEFF",
            "surface.elevated": "#123943",
            "feedback.warning.on_container": "#F1FEFF",
            "feedback.info.on_container": "#F1FEFF",
            "feedback.danger.base": "#B83A52",
            "feedback.danger.on_container": "#F1FEFF",
            "surface.primary": "#0D2931",
            "feedback.warning.content": "#F1FEFF",
            "disabled.surface": "#123943",
            "accent.on_primary": "#061215",
            "selection.border": "#1FB3FF",
            "accent.primary": "#1FB3FF",
            "accent.on_secondary": "#061215",
            "canvas": "#071A20",
            "border.strong": "#F1FEFF",
            "content.secondary": "#B8DADF",
            "feedback.info.content": "#F1FEFF",
            "border.subtle": "#2A5962",
            "feedback.info.container": "#123B45",
            "selection.content": "#061215",
            "feedback.warning.base": "#B97811",
            "feedback.info.base": "#1FB3FF",
            "content.muted": "#7DA8AE",
            "accent.secondary": "#1FB3FF",
            "feedback.success.base": "#2AAE83",
            "surface.secondary": "#071A20",
            "border.default": "#B8DADF",
            "feedback.danger.container": "#123B45"
          },
          "contrast": [
            {
              "valid": true,
              "background": "canvas",
              "foreground": "content.primary",
              "ratio": 17.29
            },
            {
              "valid": true,
              "background": "surface.primary",
              "foreground": "content.primary",
              "ratio": 14.77
            },
            {
              "valid": true,
              "background": "canvas",
              "foreground": "content.secondary",
              "ratio": 12.0
            },
            {
              "valid": true,
              "background": "canvas",
              "foreground": "content.muted",
              "ratio": 6.86
            },
            {
              "valid": true,
              "background": "accent.primary",
              "foreground": "accent.on_primary",
              "ratio": 8.1
            },
            {
              "valid": true,
              "background": "accent.secondary",
              "foreground": "accent.on_secondary",
              "ratio": 8.1
            },
            {
              "valid": true,
              "background": "feedback.info.container",
              "foreground": "feedback.info.on_container",
              "ratio": 11.72
            },
            {
              "valid": true,
              "background": "feedback.success.container",
              "foreground": "feedback.success.on_container",
              "ratio": 11.72
            },
            {
              "valid": true,
              "background": "feedback.warning.container",
              "foreground": "feedback.warning.on_container",
              "ratio": 11.72
            },
            {
              "valid": true,
              "background": "feedback.danger.container",
              "foreground": "feedback.danger.on_container",
              "ratio": 11.72
            },
            {
              "valid": true,
              "background": "selection.fill",
              "foreground": "selection.content",
              "ratio": 8.1
            }
          ],
          "customColors": {
            "identity.oracle_blue": "#1FB3FF"
          }
        }
      },
      "spacing": {
        "lg": 24.0,
        "md": 16.0,
        "sm": 8.0
      },
      "radii": {
        "standard": 8.0
      },
      "shadows": {
        "low": {
          "x": 0.0,
          "blur": 10.0,
          "color": "#061215",
          "spread": 0.0,
          "y": 2.0
        }
      },
      "typography": {
        "body": {
          "size": 16.0,
          "style": "normal",
          "weight": 400,
          "fontFamily": "primary",
          "letterSpacing": 0.0,
          "lineHeight": 1.5
        },
        "code": {
          "size": 14.0,
          "style": "normal",
          "weight": 400,
          "fontFamily": "primary",
          "letterSpacing": 0.0,
          "lineHeight": 1.5
        },
        "display": {
          "size": 40.0,
          "style": "normal",
          "weight": 700,
          "fontFamily": "primary",
          "letterSpacing": 0.0,
          "lineHeight": 1.1
        },
        "heading": {
          "size": 30.0,
          "style": "normal",
          "weight": 700,
          "fontFamily": "primary",
          "letterSpacing": 0.0,
          "lineHeight": 1.2
        },
        "label": {
          "size": 14.0,
          "style": "normal",
          "weight": 700,
          "fontFamily": "primary",
          "letterSpacing": 0.0,
          "lineHeight": 1.4
        },
        "title": {
          "size": 22.0,
          "style": "normal",
          "weight": 700,
          "fontFamily": "primary",
          "letterSpacing": 0.0,
          "lineHeight": 1.3
        }
      },
      "borders": {
        "standard": 1.0
      }
    }
  ],
  "assets": [],
  "defaultAppearance": "dark",
  "defaultTheme": "sibyl",
  "fonts": [
    {
      "id": "primary",
      "name": "primary",
      "faces": [
        {
          "asset": "fonts/primary.ttf",
          "style": "normal",
          "weight": 400
        }
      ]
    }
  ]
};
