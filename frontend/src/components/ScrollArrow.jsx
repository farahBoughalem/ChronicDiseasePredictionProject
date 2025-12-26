import "../styles/ScrollArrow.css";

export default function ScrollArrow({ href = "#content", label = "Scroll down" }) {
  return (
    <a className="scroll-arrow" href={href} aria-label={label}>
      <span className="chevron" />
      <span className="chevron" />
      <span className="chevron" />
    </a>
  );
}
