import { useEffect, useId, useRef, useState } from "react";
import { SYSTEM_LABELS, describeError, getExamples } from "../api.js";
import { canRun, parseSkills } from "../capabilities.js";
import { EXAMPLE_POOL_LIMIT, pickExamples } from "../examplePick.js";
import Notice from "./Notice.jsx";

const METHODS = [
  { id: "paste", kind: "text", label: "Paste text" },
  { id: "form", kind: "form", label: "Fill in a form" },
  { id: "example", kind: "corpus", label: "Pick an example" },
];
const K_CHOICES = [5, 10, 20];
const EMPTY_MESSAGES = {
  paste: "Paste some text first.",
  form: "Enter a job title or at least one skill.",
  example: "Choose an example first.",
};

function Required() {
  return (
    <>
      <span className="required-dot" aria-hidden="true" />
      <span className="sr-only">required</span>
    </>
  );
}

function Field({ label, children, hint, required = false, errorId }) {
  const id = useId();
  const hintId = `${id}-hint`;
  const describedBy = [hint && hintId, errorId].filter(Boolean).join(" ") || undefined;
  return (
    <div className="field">
      <label htmlFor={id}>
        {label}
        {required && <Required />}
      </label>
      {children(id, describedBy)}
      {hint && <span id={hintId}>{hint}</span>}
    </div>
  );
}

function readFileText(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(reader.error);
    reader.readAsText(file);
  });
}

function ExampleOption({ item }) {
  return (
    <option value={item.id}>
      {item.title || "Untitled"}, {item.level}, {item.skills.length} skills
    </option>
  );
}

function ExampleOptions({ picked }) {
  const options = (items) => items.map((item) => <ExampleOption key={item.id} item={item} />);
  if (picked.random.length === 0) return options(picked.defaults);
  return (
    <>
      <optgroup label="Suggested examples">{options(picked.defaults)}</optgroup>
      <optgroup label="More examples, shuffled">{options(picked.random)}</optgroup>
    </>
  );
}

function ExamplePicker({ side, fetchImpl, chosen, onChoose, random }) {
  const [state, setState] = useState({ items: null, picked: null, error: null });
  const randomRef = useRef(random);
  const chosenRef = useRef(chosen);
  randomRef.current = random;
  chosenRef.current = chosen;
  useEffect(() => {
    let live = true;
    getExamples(side, EXAMPLE_POOL_LIMIT, fetchImpl).then(
      (data) =>
        live &&
        setState({
          items: data.items,
          picked: pickExamples(data.items, { random: randomRef.current, chosenId: chosenRef.current }),
          error: null,
        }),
      (error) => live && setState({ items: null, picked: null, error }),
    );
    return () => {
      live = false;
    };
  }, [side, fetchImpl]);

  if (state.error) {
    return (
      <div className="field">
        <Notice kind="error" role="alert">{describeError(state.error).title}</Notice>
      </div>
    );
  }
  if (!state.items) {
    return (
      <div className="field">
        <p role="status">Loading examples…</p>
      </div>
    );
  }
  return (
    <Field label={`Example ${side}`} required>
      {(id) => (
        <select id={id} value={chosen ?? ""} onChange={(event) => onChoose(event.target.value || null)}>
          <option value="">Choose an example</option>
          <ExampleOptions picked={state.picked} />
        </select>
      )}
    </Field>
  );
}

function PastePanel({ isResume, fields, set, update, capabilities, errorId }) {
  const [fileError, setFileError] = useState(null);
  async function loadFile(event) {
    const file = event.target.files[0];
    if (!file) return;
    try {
      update({ text: await readFileText(file) });
      setFileError(null);
    } catch {
      setFileError("That file could not be read.");
    }
  }
  const { detail } = capabilities.structure;
  return (
    <>
      {!isResume && (
        <Field label="Job title (optional)">
          {(id) => <input id={id} type="text" value={fields.pasteTitle} onChange={set("pasteTitle")} />}
        </Field>
      )}
      <Field label={isResume ? "Resume text" : "Job posting text"} required errorId={errorId}>
        {(id, describedBy) => (
          <textarea id={id} rows={10} aria-describedby={describedBy} value={fields.text} onChange={set("text")} />
        )}
      </Field>
      {detail && <p>Text you paste is sent to {detail} to be read.</p>}
      <Field label="Load a .txt or .md file">
        {(id) => <input id={id} type="file" accept=".txt,.md,text/plain,text/markdown" onChange={loadFile} />}
      </Field>
      {fileError && <Notice kind="error" role="alert">{fileError}</Notice>}
    </>
  );
}

function FormPanel({ isResume, levels, fields, set, errorId }) {
  const pickable = levels.filter((level) => level !== "Not specified");
  return (
    <>
      <Field label="Job title" hint="Add a title, skills, or both" errorId={errorId}>
        {(id, describedBy) => (
          <input id={id} type="text" aria-describedby={describedBy} value={fields.title} onChange={set("title")} />
        )}
      </Field>
      {isResume ? (
        <Field label="Years of experience">
          {(id) => (
            <input id={id} type="number" min="0" max="60" step="any" value={fields.years} onChange={set("years")} />
          )}
        </Field>
      ) : (
        <Field label="Experience level">
          {(id) => (
            <select id={id} value={fields.level} onChange={set("level")}>
              <option value="">Not specified</option>
              {pickable.map((level) => (
                <option key={level} value={level}>{level}</option>
              ))}
            </select>
          )}
        </Field>
      )}
      <Field label="Skills" hint="One per line or comma-separated">
        {(id, describedBy) => (
          <textarea id={id} rows={5} aria-describedby={describedBy} value={fields.skills} onChange={set("skills")} />
        )}
      </Field>
    </>
  );
}

function buildFormQuery({ fields, isResume }) {
  const skills = parseSkills(fields.skills);
  const title = fields.title.trim();
  if (!title && skills.length === 0) return null;
  const query = { kind: "form", title };
  if (isResume && fields.years !== "") query.years_experience = Number(fields.years);
  if (!isResume && fields.level) query.level = fields.level;
  return { ...query, skills };
}

function buildQuery({ methodId, fields, isResume }) {
  if (methodId === "paste") {
    if (!fields.text.trim()) return null;
    const title = fields.pasteTitle.trim();
    return { kind: "text", text: fields.text, ...(!isResume && title ? { title } : {}) };
  }
  if (methodId === "form") return buildFormQuery({ fields, isResume });
  return fields.example ? { kind: "corpus", id: fields.example } : null;
}

const INITIAL_FIELDS = { text: "", pasteTitle: "", title: "", years: "", level: "", skills: "", example: null };
const EXAMPLE_METHOD = METHODS[2];
const ERROR_ID = "query-form-error";

export default function QueryForm({ direction, health, busy, onSubmit, fetchImpl, random = Math.random }) {
  const isResume = direction === "resume_to_posting";
  const side = isResume ? "resume" : "posting";
  const { capabilities, levels } = health;
  const [system, setSystem] = useState("Full");
  const [k, setK] = useState(10);
  const [fields, setFields] = useState(INITIAL_FIELDS);
  const [error, setError] = useState(null);
  const verdict = (method) => canRun(method.kind, system, capabilities);
  const [methodId, setMethodId] = useState(
    () => (METHODS.find((m) => canRun(m.kind, "Full", capabilities).ok) ?? EXAMPLE_METHOD).id,
  );
  const update = (patch) => setFields((current) => ({ ...current, ...patch }));
  const set = (name) => (event) => update({ [name]: event.target.value });
  const errorId = error ? ERROR_ID : undefined;

  function submit(event) {
    event.preventDefault();
    const verdictNow = verdict(METHODS.find((m) => m.id === methodId));
    if (!verdictNow.ok) return setError(verdictNow.reason);
    const query = buildQuery({ methodId, fields, isResume });
    if (!query) return setError(EMPTY_MESSAGES[methodId]);
    setError(null);
    onSubmit({ direction, k, system, query });
  }

  return (
    <form onSubmit={submit} noValidate>
      <div className="segmented" role="group" aria-label="How to provide the query">
        {METHODS.map((method) => {
          const { ok, reason } = verdict(method);
          const reasonId = `method-${method.id}-reason`;
          return (
            <span key={method.id}>
              <button
                type="button"
                className="btn"
                aria-pressed={methodId === method.id}
                aria-describedby={ok ? undefined : reasonId}
                disabled={!ok}
                onClick={() => setMethodId(method.id)}
              >
                {method.label}
              </button>
              {!ok && <span id={reasonId}> {reason}</span>}
            </span>
          );
        })}
      </div>

      {methodId === "paste" && (
        <PastePanel
          isResume={isResume} fields={fields} set={set} update={update}
          capabilities={capabilities} errorId={errorId}
        />
      )}
      {methodId === "form" && (
        <FormPanel isResume={isResume} levels={levels} fields={fields} set={set} errorId={errorId} />
      )}
      <div className="control-row">
        {methodId === "example" && (
          <ExamplePicker
            side={side} fetchImpl={fetchImpl} chosen={fields.example} random={random}
            onChoose={(example) => update({ example })}
          />
        )}
        <Field label="Number of results">
          {(id) => (
            <select id={id} value={k} onChange={(event) => setK(Number(event.target.value))}>
              {K_CHOICES.map((choice) => <option key={choice} value={choice}>{choice}</option>)}
            </select>
          )}
        </Field>
        <Field label="Ranking method">
          {(id) => (
            <select id={id} value={system} onChange={(event) => setSystem(event.target.value)}>
              {Object.entries(SYSTEM_LABELS).map(([value, label]) => (
                <option key={value} value={value}>{label}</option>
              ))}
            </select>
          )}
        </Field>
      </div>

      {error && <Notice kind="error" role="alert" id={ERROR_ID}>{error}</Notice>}
      <button type="submit" className="btn btn-cta" disabled={busy}>
        {isResume ? "Find jobs" : "Find candidates"}
      </button>
    </form>
  );
}
