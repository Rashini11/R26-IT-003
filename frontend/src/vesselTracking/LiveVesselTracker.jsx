import { useEffect, useState } from "react";
import { ArrowDown, Navigation, Ship } from "lucide-react";
import "./LiveVesselTracker.css";

const DEFAULT_LOCATION = {
  latitude: 6.9514,
  longitude: 79.8478,
  source: "default",
};

const VESSELFINDER_MAP_HEIGHT = 600;
const VESSELFINDER_HEADER_HEIGHT = 20;

function createVesselFinderDocument({ latitude, longitude, zoom }) {
  return `<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
      html, body { width: 100%; height: 100%; margin: 0; overflow: hidden; }
      body { position: relative; }
      #vesselfinder {
        position: absolute;
        top: -${VESSELFINDER_HEADER_HEIGHT}px;
        left: 0;
        width: 100%;
        height: ${VESSELFINDER_MAP_HEIGHT + VESSELFINDER_HEADER_HEIGHT}px;
        border: 0;
      }
    </style>
  </head>
  <body>
    <script>
      var width = "100%";
      var height = "${VESSELFINDER_MAP_HEIGHT + VESSELFINDER_HEADER_HEIGHT}";
      var latitude = "${latitude}";
      var longitude = "${longitude}";
      var zoom = "${zoom}";
      var names = false;
      var show_track = false;
    </script>
    <script src="https://www.vesselfinder.com/aismap.js"></script>
  </body>
</html>`;
}

export default function LiveVesselTracker({ onGoToImageDetection }) {
  const [locationMode, setLocationMode] = useState("device");
  const [location, setLocation] = useState(DEFAULT_LOCATION);
  const [latitudeInput, setLatitudeInput] = useState(
    String(DEFAULT_LOCATION.latitude),
  );
  const [longitudeInput, setLongitudeInput] = useState(
    String(DEFAULT_LOCATION.longitude),
  );
  const [zoomInput, setZoomInput] = useState("9");
  const [mapZoom, setMapZoom] = useState(9);
  const [locationMessage, setLocationMessage] = useState(
    "Showing vessels near Colombo. Allow location access or enter another position.",
  );
  const [gpsRequest, setGpsRequest] = useState(0);
  const [gpsAvailable] = useState(
    () => typeof navigator !== "undefined" && Boolean(navigator.geolocation),
  );

  useEffect(() => {
    if (locationMode !== "device" || !gpsAvailable) return undefined;

    const applyDevicePosition = (position) => {
      const nextLocation = {
        latitude: position.coords.latitude,
        longitude: position.coords.longitude,
        source: "device",
      };
      setLocation(nextLocation);
      setLatitudeInput(nextLocation.latitude.toFixed(6));
      setLongitudeInput(nextLocation.longitude.toFixed(6));
      setLocationMessage("Map centered on your device location.");
    };
    const handleError = (geoError) => {
      setLocationMessage(
        geoError.code === geoError.PERMISSION_DENIED
          ? "Location permission was denied. The map remains centered near Colombo; enter a position manually."
          : "Device location is unavailable. The map remains centered near Colombo; enter a position manually.",
      );
    };

    navigator.geolocation.getCurrentPosition(
      applyDevicePosition,
      handleError,
      { enableHighAccuracy: true, maximumAge: 5000, timeout: 15000 },
    );
    const watchId = navigator.geolocation.watchPosition(
      applyDevicePosition,
      handleError,
      { enableHighAccuracy: true, maximumAge: 5000, timeout: 20000 },
    );
    return () => navigator.geolocation.clearWatch(watchId);
  }, [gpsAvailable, gpsRequest, locationMode]);

  const useDeviceGps = () => {
    setLocationMode("device");
    setGpsRequest((attempt) => attempt + 1);
    setLocationMessage("Requesting device location permission…");
  };

  const saveManualLocation = (event) => {
    event.preventDefault();
    const latitude = Number(latitudeInput);
    const longitude = Number(longitudeInput);
    const zoom = Number(zoomInput);
    if (
      !Number.isFinite(latitude)
      || !Number.isFinite(longitude)
      || latitude < -90
      || latitude > 90
      || longitude < -180
      || longitude > 180
      || !Number.isInteger(zoom)
      || zoom < 3
      || zoom > 18
    ) {
      setLocationMessage(
        "Enter valid coordinates and a whole-number zoom from 3 to 18.",
      );
      return;
    }

    const coordinatesChanged =
      latitude !== location.latitude || longitude !== location.longitude;
    const source = coordinatesChanged ? "manual" : location.source;
    if (coordinatesChanged) setLocationMode("manual");
    setLocation({ latitude, longitude, source });
    setMapZoom(zoom);
    setLocationMessage("VesselFinder map settings updated.");
  };
  const mapCenter = {
    latitude: Number(location.latitude.toFixed(4)),
    longitude: Number(location.longitude.toFixed(4)),
  };
  const vesselFinderMapSettings = { ...mapCenter, zoom: mapZoom };

  return (
    <section className="live-vessel-tracker">
      <div className="lvt-control-sections">
        <section className="lvt-control-card lvt-intro-card">
          <div className="lvt-card-kicker">VESSEL TRACKING</div>
          <h2 className="lvt-main-title">
            <Ship size={17} />
            YOUR LOCATION · MAP
          </h2>
          <p>Choose a position and zoom level to explore vessel activity.</p>
          <button
            type="button"
            className="lvt-image-detection-button"
            onClick={onGoToImageDetection}
          >
            <Ship size={14} />
            IMAGE DETECTION (YOLO)
            <ArrowDown size={13} />
          </button>
        </section>

        <section className="lvt-control-card lvt-location-settings-card">
          <div className="lvt-combined-card-top">
            <div>
              <div className="lvt-card-kicker">CURRENT POSITION</div>
              <div className="lvt-current-coordinates">
                <Navigation size={15} />
                <span>
                  {location.latitude.toFixed(5)}, {location.longitude.toFixed(5)}
                </span>
              </div>
              <p className="lvt-location-source">Location source: {location.source}</p>
            </div>
            <button
              type="button"
              className={locationMode === "device" ? "is-selected" : ""}
              onClick={useDeviceGps}
            >
              <Navigation size={13} /> Use device GPS
            </button>
          </div>
          <div className="lvt-card-kicker">MAP SETTINGS</div>
          <form className="lvt-manual-form" onSubmit={saveManualLocation}>
            <label>
              Latitude
              <input
                type="number"
                min="-90"
                max="90"
                step="any"
                value={latitudeInput}
                onChange={(event) => setLatitudeInput(event.target.value)}
                placeholder="e.g. 6.9514"
                required
              />
            </label>
            <label>
              Longitude
              <input
                type="number"
                min="-180"
                max="180"
                step="any"
                value={longitudeInput}
                onChange={(event) => setLongitudeInput(event.target.value)}
                placeholder="e.g. 79.8478"
                required
              />
            </label>
            <label>
              Map zoom (3–18)
              <input
                type="number"
                min="3"
                max="18"
                step="1"
                value={zoomInput}
                onChange={(event) => setZoomInput(event.target.value)}
                required
              />
            </label>
            <button type="submit">Apply map settings</button>
          </form>
          <p className="lvt-location-message" aria-live="polite">
            {locationMessage}
          </p>
        </section>
      </div>

      <div className="lvt-map-wrap">
        <div className="lvt-map-title">
          <span>VESSELFINDER · LIVE VESSELS</span>
          <span>Latitude {mapCenter.latitude} · Longitude {mapCenter.longitude} · Zoom {mapZoom}</span>
        </div>
        <iframe
          key={`vesselfinder-${mapCenter.latitude},${mapCenter.longitude},${mapZoom}`}
          className="lvt-vesselfinder-map"
          title="VesselFinder embedded live vessel map"
          srcDoc={createVesselFinderDocument(vesselFinderMapSettings)}
          loading="lazy"
          referrerPolicy="strict-origin-when-cross-origin"
        />
      </div>
      <p className="lvt-disclaimer">
        Map and vessel information are provided by VesselFinder and are subject
        to its availability and terms.
      </p>
    </section>
  );
}
