const ARRAY_PROPERTY_HINTS = [
  'predictions',
  'prediction',
  'results',
  'output',
  'outputs',
  'values',
  'value',
  'data'
];

const PRIMARY_KEY_HINTS = [
  'next_app',
  'predicted_app',
  'app',
  'application',
  'target',
  'label'
];

const DAY_KEY_HINTS = ['weekday', 'day', 'date'];
const TIME_KEY_HINTS = ['hour', 'time', 'timestamp'];

const isPlainObject = (value) =>
  value !== null && typeof value === 'object' && !Array.isArray(value);

const tryParseJson = (value) => {
  if (typeof value !== 'string') {
    return value;
  }
  const trimmed = value.trim();
  if (!trimmed) {
    return value;
  }
  if ((trimmed.startsWith('{') && trimmed.endsWith('}')) ||
    (trimmed.startsWith('[') && trimmed.endsWith(']'))) {
    try {
      return JSON.parse(trimmed);
    } catch {
      return value;
    }
  }
  return value;
};

const pickKeyByHints = (keys, hints, exclude = new Set()) => {
  const available = keys.filter((key) => !exclude.has(key));
  const lowerKeyMap = new Map(
    available.map((key) => [key.toLowerCase(), key])
  );

  for (const hint of hints) {
    const exact = lowerKeyMap.get(hint);
    if (exact) {
      return exact;
    }
  }

  for (const hint of hints) {
    const partial = available.find((key) => key.toLowerCase().includes(hint));
    if (partial) {
      return partial;
    }
  }

  return undefined;
};

const titleCase = (value) =>
  value.replace(/[_\s]+/g, ' ')
    .replace(/\b\w/g, (char) => char.toUpperCase());

const formatValue = (value) => {
  if (value === null || value === undefined) {
    return '';
  }
  if (typeof value === 'string') {
    return value;
  }
  if (typeof value === 'number') {
    return Number.isFinite(value) ? value.toString() : '';
  }
  if (typeof value === 'boolean') {
    return value ? 'Yes' : 'No';
  }
  if (value instanceof Date) {
    return value.toISOString();
  }
  if (Array.isArray(value)) {
    return value.map(formatValue).join(', ');
  }
  return JSON.stringify(value);
};

const findEntries = (raw) => {
  const parsed = tryParseJson(raw);

  if (parsed === null || parsed === undefined) {
    return [];
  }

  if (Array.isArray(parsed)) {
    return parsed;
  }

  if (isPlainObject(parsed)) {
    for (const key of ARRAY_PROPERTY_HINTS) {
      if (key in parsed) {
        const candidate = tryParseJson(parsed[key]);
        if (Array.isArray(candidate)) {
          return candidate;
        }
      }
    }
    return [parsed];
  }

  return [parsed];
};

const buildMetaEntries = (entry, usedKeys = new Set()) =>
  Object.entries(entry)
    .filter(([key]) => !usedKeys.has(key))
    .map(([key, value]) => ({
      label: titleCase(key),
      value: formatValue(value)
    }))
    .filter(({ value }) => value !== '');

const deriveSubtitle = (entry, dayKey, primaryKey, meta) => {
  if (dayKey) {
    const dayValue = formatValue(entry[dayKey]);
    if (dayValue) {
      return dayValue;
    }
  }

  if (meta.length > 0) {
    const [firstMeta] = meta;
    if (firstMeta?.value) {
      meta.splice(0, 1);
      return `${firstMeta.label}: ${firstMeta.value}`;
    }
  }

  if (primaryKey) {
    return titleCase(primaryKey);
  }

  return undefined;
};

const normalizeEntry = (entry, index) => {
  const parsedEntry = tryParseJson(entry);
  const rawText = isPlainObject(parsedEntry) || Array.isArray(parsedEntry)
    ? JSON.stringify(parsedEntry, null, 2)
    : formatValue(parsedEntry);

  if (!isPlainObject(parsedEntry)) {
    return {
      id: `prediction-${index}`,
      title: formatValue(parsedEntry) || 'Unknown Prediction',
      subtitle: undefined,
      time: undefined,
      meta: [],
      raw: entry,
      rawText
    };
  }

  const keys = Object.keys(parsedEntry);
  const usedKeys = new Set();

  const primaryKey = pickKeyByHints(keys, PRIMARY_KEY_HINTS);
  if (primaryKey) {
    usedKeys.add(primaryKey);
  }

  const timeKey = pickKeyByHints(keys, TIME_KEY_HINTS, usedKeys);
  if (timeKey) {
    usedKeys.add(timeKey);
  }

  const dayKey = pickKeyByHints(keys, DAY_KEY_HINTS, usedKeys);
  if (dayKey) {
    usedKeys.add(dayKey);
  }

  const title = primaryKey
    ? formatValue(parsedEntry[primaryKey]) || 'Unknown Prediction'
    : keys.length > 0
      ? formatValue(parsedEntry[keys[0]]) || 'Unknown Prediction'
      : 'Unknown Prediction';

  const meta = buildMetaEntries(parsedEntry, usedKeys);
  const subtitle = deriveSubtitle(parsedEntry, dayKey, primaryKey, meta);

  return {
    id: `prediction-${index}`,
    title,
    subtitle,
    time: timeKey ? formatValue(parsedEntry[timeKey]) : undefined,
    meta,
    raw: entry,
    rawText
  };
};

export const normalizePredictions = (predictionData) => {
  const entries = findEntries(predictionData);
  return entries.map((entry, index) => normalizeEntry(entry, index));
};
