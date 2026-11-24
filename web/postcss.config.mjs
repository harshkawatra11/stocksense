import prefixer from "postcss-prefix-selector";

// Scopes each Market Lab stylesheet under its section class (replaces Market Lab's Shadow DOM).
const config = {
  plugins: [
    prefixer({
      prefix: "",
      transform(prefix, selector, prefixed, file) {
        const m = /ml-(smart|sector|lab)\.css$/.exec(file || "");
        if (!m) return selector;
        const scope = `.ml-${m[1]}`;
        if (selector === ":root" || selector === "html,body" || selector === "html" || selector === "body") return scope;
        if (selector.startsWith("@")) return selector;
        return `${scope} ${selector}`;
      },
    }),
  ],
};

export default config;
