import { useRef } from "react";
import {
  Anchor,
  Waves,
  Ship,
  RadioTower,
  MessageSquare,
  Navigation,
  Layers,
  ArrowRight,
  Image as ImageIcon,
  Route,
  ShieldAlert,
  Compass,
  ScanSearch,
  ChartColumnIncreasing,
} from "lucide-react";
import "./HomePage.css";

const FULL_TITLE =
  "OceanIQ: AI-Driven Multi-Sensor Decision Support System for Autonomous Ship Surveillance and Condition Assessment.";

export default function HomePage({ onNavigate }) {
  const componentsRef = useRef(null);

  const go = (moduleId) => {
    if (typeof onNavigate === "function") onNavigate(moduleId);
  };

  const scrollToComponents = () => {
    const reduceMotion =
      typeof window !== "undefined" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    componentsRef.current?.scrollIntoView({
      behavior: reduceMotion ? "auto" : "smooth",
      block: "start",
    });
  };

  return (
    <div className="home-page">
      <section className="home-hero" aria-labelledby="home-hero-heading">
        <div className="home-hero-copy">
          <p className="home-eyebrow">Maritime research · R26-IT-003</p>
          <h1 id="home-hero-heading">Maritime insight, brought together.</h1>
          <p className="home-lead">
            Explore sea conditions, vessel detections, radar contacts and hull condition through one
            connected research workspace.
          </p>
          <div className="home-hero-actions">
            <button type="button" className="home-btn home-btn--primary" onClick={() => go("chat")}>
              Open Chat Assistant
              <ArrowRight size={16} aria-hidden="true" />
            </button>
            <button type="button" className="home-btn home-btn--ghost" onClick={scrollToComponents}>
              Explore components
            </button>
          </div>
          <p className="home-title-line">{FULL_TITLE}</p>
        </div>

        <div className="home-hero-visual" aria-hidden="true">
          <div className="home-overview-card">
            <p className="home-eyebrow">System overview</p>
            <svg className="home-overview-svg" viewBox="0 0 320 240" role="presentation">
              <defs>
                <radialGradient id="homeRadarGlow" cx="50%" cy="50%" r="50%">
                  <stop offset="0%" stopColor="rgba(102,214,255,0.35)" />
                  <stop offset="100%" stopColor="rgba(102,214,255,0)" />
                </radialGradient>
              </defs>
              <circle cx="160" cy="120" r="88" fill="url(#homeRadarGlow)" />
              <circle cx="160" cy="120" r="70" fill="none" stroke="#295676" strokeWidth="1" />
              <circle cx="160" cy="120" r="46" fill="none" stroke="#1f3b56" strokeWidth="1" />
              <circle cx="160" cy="120" r="22" fill="none" stroke="#39728d" strokeWidth="1" />
              <line x1="160" y1="120" x2="220" y2="72" stroke="#66d6ff" strokeWidth="1.5" />
              <circle cx="160" cy="120" r="4" fill="#66d6ff" />
              <rect x="24" y="28" width="78" height="34" rx="10" fill="#10263b" stroke="#1f3b56" />
              <text x="36" y="50" fill="#7f9ab4" fontSize="11">Hull</text>
              <rect x="218" y="28" width="78" height="34" rx="10" fill="#10263b" stroke="#1f3b56" />
              <text x="232" y="50" fill="#7f9ab4" fontSize="11">Sea state</text>
              <rect x="24" y="178" width="78" height="34" rx="10" fill="#10263b" stroke="#1f3b56" />
              <text x="34" y="200" fill="#7f9ab4" fontSize="11">Radar</text>
              <rect x="218" y="178" width="78" height="34" rx="10" fill="#10263b" stroke="#1f3b56" />
              <text x="230" y="200" fill="#7f9ab4" fontSize="11">Vessel</text>
              <rect x="112" y="98" width="96" height="44" rx="12" fill="#15334b" stroke="#66d6ff" />
              <text x="128" y="118" fill="#e8f4ff" fontSize="12" fontWeight="700">OceanIQ</text>
              <text x="124" y="134" fill="#72e5b1" fontSize="10">workspace</text>
            </svg>
            <p className="home-overview-caption">
              Conceptual component map — not live sensor data.
            </p>
          </div>
        </div>
      </section>

      <section className="home-section" aria-labelledby="home-purpose-heading">
        <div className="home-section-head">
          <p className="home-eyebrow">Purpose</p>
          <h2 id="home-purpose-heading">What OceanIQ helps you do</h2>
        </div>
        <p className="home-section-copy">
          Maritime information often comes from separate sources. OceanIQ brings selected
          image-analysis and movement-analysis workflows into a shared interface, helping users
          inspect results, understand uncertainty and review available prediction records.
        </p>
        <div className="home-benefit-grid">
          <article className="home-benefit">
            <ScanSearch size={18} aria-hidden="true" />
            <h3>Analyse maritime imagery</h3>
            <p>Use dedicated research models for hull, sea-state, vessel and radar imagery.</p>
          </article>
          <article className="home-benefit">
            <ChartColumnIncreasing size={18} aria-hidden="true" />
            <h3>Interpret with context</h3>
            <p>Review scores, visualisations and available explanations for each result.</p>
          </article>
          <article className="home-benefit">
            <Compass size={18} aria-hidden="true" />
            <h3>Explore movement risk</h3>
            <p>
              Review movement forecasts and encounter-risk measures when the required AIS inputs
              are available.
            </p>
          </article>
        </div>
      </section>

      <section
        className="home-section"
        aria-labelledby="home-components-heading"
        ref={componentsRef}
        id="home-components"
      >
        <div className="home-section-head">
          <p className="home-eyebrow">Research components</p>
          <h2 id="home-components-heading">Four image-analysis modules</h2>
        </div>
        <div className="home-component-grid">
          <article className="home-component-card">
            <div className="home-component-icon" aria-hidden="true">
              <Anchor size={20} />
            </div>
            <h3>Hull Condition Assessment</h3>
            <p>
              Analyse hull inspection images to assess the model’s supported condition categories
              and review available visual explanations.
            </p>
            <dl className="home-io">
              <div>
                <dt>Input</dt>
                <dd>Hull inspection image</dd>
              </div>
              <div>
                <dt>Output</dt>
                <dd>Condition or rejection result, confidence and Grad-CAM where available</dd>
              </div>
            </dl>
            <button type="button" className="home-btn home-btn--ghost" onClick={() => go("hull")}>
              Inspect hull
            </button>
          </article>

          <article className="home-component-card">
            <div className="home-component-icon" aria-hidden="true">
              <Waves size={20} />
            </div>
            <h3>Sea-State Classification</h3>
            <p>
              Analyse sea-surface images to classify supported conditions and review confidence,
              uncertainty and available recommendations.
            </p>
            <dl className="home-io">
              <div>
                <dt>Input</dt>
                <dd>Sea-surface image</dd>
              </div>
              <div>
                <dt>Supported classes</dt>
                <dd>
                  Calm, Moderate, Rough and Very Rough, subject to the active label map
                </dd>
              </div>
              <div>
                <dt>Output</dt>
                <dd>Class, model score and available validation information</dd>
              </div>
            </dl>
            <button type="button" className="home-btn home-btn--ghost" onClick={() => go("sea")}>
              Analyse sea state
            </button>
          </article>

          <article className="home-component-card">
            <div className="home-component-icon" aria-hidden="true">
              <Ship size={20} />
            </div>
            <h3>Drone-Assisted Vessel Detection</h3>
            <p>
              Locate and classify visible vessels in imagery using the existing detection pipeline.
              Image and video uploads are supported in the current interface.
            </p>
            <dl className="home-io">
              <div>
                <dt>Input</dt>
                <dd>Vessel image or video</dd>
              </div>
              <div>
                <dt>Labels</dt>
                <dd>Local/Foreign Boat or Ship, with SL Flag cues when detected</dd>
              </div>
              <div>
                <dt>Output</dt>
                <dd>Detections, scores and annotated results</dd>
              </div>
            </dl>
            <button type="button" className="home-btn home-btn--ghost" onClick={() => go("boat")}>
              Detect vessels
            </button>
          </article>

          <article className="home-component-card">
            <div className="home-component-icon" aria-hidden="true">
              <RadioTower size={20} />
            </div>
            <h3>Radar Object Classification</h3>
            <p>
              Classify uploaded radar imagery as bird, ship or unknown using the active three-class
              model.
            </p>
            <dl className="home-io">
              <div>
                <dt>Input</dt>
                <dd>Radar image</dd>
              </div>
              <div>
                <dt>Output</dt>
                <dd>Predicted class and bird, ship and unknown probabilities</dd>
              </div>
            </dl>
            <button type="button" className="home-btn home-btn--ghost" onClick={() => go("radar")}>
              Classify radar
            </button>
          </article>
        </div>
      </section>

      <section className="home-section" aria-labelledby="home-ways-heading">
        <div className="home-section-head">
          <p className="home-eyebrow">Workflows</p>
          <h2 id="home-ways-heading">Two ways to analyse</h2>
        </div>
        <div className="home-ways-grid">
          <article className="home-way-card">
            <ImageIcon size={18} aria-hidden="true" />
            <h3>Manual analysis</h3>
            <p>Choose a component directly, upload a suitable image and inspect its detailed results.</p>
            <button type="button" className="home-btn home-btn--ghost" onClick={scrollToComponents}>
              Explore components
            </button>
          </article>
          <article className="home-way-card">
            <MessageSquare size={18} aria-hidden="true" />
            <h3>Chat Assistant</h3>
            <p>
              Upload an image, select an analysis type and receive the component’s results as
              readable messages with available charts or annotated images.
            </p>
            <p className="home-note">
              The assistant uses predefined conversation logic and existing model outputs. It does
              not use an LLM. Automatic image-type routing is not available in this version.
            </p>
            <button type="button" className="home-btn home-btn--primary" onClick={() => go("chat")}>
              Open Chat Assistant
            </button>
          </article>
        </div>
      </section>

      <section className="home-section home-section--sim" aria-labelledby="home-sim-heading">
        <div className="home-section-head">
          <p className="home-eyebrow">Movement analysis</p>
          <h2 id="home-sim-heading">AIS and encounter-risk workspace</h2>
        </div>
        <div className="home-sim-card">
          <div>
            <Navigation size={20} aria-hidden="true" />
            <p>
              Explore vessel movement using prepared Automatic Identification System (AIS)
              sequences, GRU-based forecasting and Distance at Closest Point of Approach (DCPA)
              and Time to Closest Point of Approach (TCPA) calculations. The simulation presents
              encounter-risk categories using configured rules.
            </p>
            <p className="home-note">
              This workflow needs the relevant runtime data and movement inputs. A radar or vessel
              photograph alone does not provide a forecast or collision-risk calculation. If
              simulation assets are missing on this machine, the destination still opens and reports
              availability truthfully.
            </p>
            <button
              type="button"
              className="home-btn home-btn--primary"
              onClick={() => go("simulation")}
            >
              Open Live Simulation
              <Route size={16} aria-hidden="true" />
            </button>
          </div>
        </div>
      </section>

      <section className="home-section" aria-labelledby="home-start-heading">
        <div className="home-section-head">
          <p className="home-eyebrow">Getting started</p>
          <h2 id="home-start-heading">How to get started</h2>
        </div>
        <ol className="home-steps">
          <li>
            <span>1</span>
            <p>Choose a component or open the Chat Assistant.</p>
          </li>
          <li>
            <span>2</span>
            <p>
              Provide a suitable image; movement analysis requires its corresponding AIS inputs.
            </p>
          </li>
          <li>
            <span>3</span>
            <p>Review the result, available visualisation and uncertainty information.</p>
          </li>
        </ol>
        <button type="button" className="home-link-btn" onClick={() => go("database")}>
          <Layers size={15} aria-hidden="true" />
          Review available records
        </button>
      </section>

      <section className="home-section home-section--limits" aria-labelledby="home-limits-heading">
        <div className="home-limits-card">
          <ShieldAlert size={18} aria-hidden="true" />
          <div>
            <h2 id="home-limits-heading">Research context and limitations</h2>
            <p>
              OceanIQ is a research prototype and decision-support workspace. Model scores express
              the model’s output and do not guarantee correctness. Results depend on input
              suitability and the limitations of each component. OceanIQ does not autonomously
              navigate vessels, replace professional judgement, or claim validated real-world safety
              guarantees.
            </p>
            <p className="home-meta">
              <strong>OceanIQ</strong>
              <br />
              {FULL_TITLE}
              <br />
              Research group R26-IT-003
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}
