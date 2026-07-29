export type CsvCell =
  | string
  | number
  | boolean
  | null
  | undefined;


function escapeCsvCell(
  value: CsvCell,
): string {
  if (
    value === null ||
    value === undefined
  ) {
    return "";
  }

  const text = String(value);

  /*
   * 只要內容包含：
   * - 逗號
   * - 雙引號
   * - 換行
   *
   * 就需要用雙引號包起來。
   */
  if (/[",\r\n]/.test(text)) {
    return `"${text.replace(
      /"/g,
      '""',
    )}"`;
  }

  return text;
}


export function downloadCsv(
  filename: string,
  headers: string[],
  rows: CsvCell[][],
): void {
  const csvContent = [
    headers,
    ...rows,
  ]
    .map((row) => {
      return row
        .map(escapeCsvCell)
        .join(",");
    })
    .join("\r\n");

  /*
   * UTF-8 BOM 可避免 Excel
   * 開啟繁體中文 CSV 時出現亂碼。
   */
  const blob = new Blob(
    [
      "\uFEFF",
      csvContent,
    ],
    {
      type: "text/csv;charset=utf-8",
    },
  );

  const objectUrl =
    URL.createObjectURL(blob);

  const link =
    document.createElement("a");

  link.href = objectUrl;
  link.download = filename;

  document.body.appendChild(link);

  link.click();
  link.remove();

  URL.revokeObjectURL(objectUrl);
}