/**
 * Converts the API's unambiguous LaTex delimiters into the syntax understood
 * by remark-math. Single-dollar parsing remains disabled, so ordinary prose
 * and currency values can never become math accidentally.
 */
export function prepareMathMarkdown(markdown: string): string {
  const fencedCodeBlock = /(^|\n)(```[\s\S]*?^```[ \t]*(?=\n|$))/gm;
  let result = "";
  let lastIndex = 0;

  for (const match of markdown.matchAll(fencedCodeBlock)) {
    const index = match.index ?? 0;
    result += convertMathDelimiters(markdown.slice(lastIndex, index));
    result += match[0];
    lastIndex = index + match[0].length;
  }

  return result + convertMathDelimiters(markdown.slice(lastIndex));
}

function convertMathDelimiters(text: string): string {
  const displayMath = text.replace(/\\\[([\s\S]*?)\\\]/g, (_, latex: string) => {
    const equation = latex.trim();
    return equation ? `\n\n$$\n${equation}\n$$\n\n` : "\\[\\]";
  });

  return displayMath.replace(/\\\(([^\n]*?)\\\)/g, (_, latex: string) => {
    const equation = latex.trim();
    return equation ? `$$${equation}$$` : "\\(\\)";
  });
}
