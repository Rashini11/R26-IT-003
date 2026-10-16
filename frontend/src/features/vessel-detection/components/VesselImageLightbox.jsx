import { useEffect } from "react";
import { X } from "lucide-react";

// Show a selected vessel image at a larger size.
export default function VesselImageLightbox({ image, onClose }) {
  useEffect(() => {
    if (!image) return undefined;
    const closeOnEscape = (event) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [image, onClose]);

  if (!image) return null;

  return (
    <div
      className="vessel-image-lightbox"
      role="presentation"
      onClick={onClose}
    >
      <div
        className="vessel-image-lightbox__content"
        role="dialog"
        aria-modal="true"
        aria-label="Expanded vessel image"
        onClick={(event) => event.stopPropagation()}
      >
        <button
          type="button"
          className="vessel-image-lightbox__close"
          onClick={onClose}
          aria-label="Close expanded image"
        >
          <X size={16} /> CLOSE
        </button>
        <img src={image.url} alt={image.alt} />
        <span>{image.alt} · CLICK OUTSIDE OR PRESS ESC TO CLOSE</span>
      </div>
    </div>
  );
}
