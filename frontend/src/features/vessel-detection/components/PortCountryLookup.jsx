import { useEffect, useRef, useState } from "react";
import { lookupVesselPortCountry } from "../api/vesselDetectionApi";

// Show the port match and let the user choose when names are ambiguous.
export default function PortCountryLookup({
  port,
  country,
  onCountryFound,
}) {
  const [lookupState, setLookupState] = useState({
    query: "",
    loading: false,
    result: null,
    error: "",
    requestId: 0,
  });
  const requestId = useRef(0);
  const controller = useRef(null);

  // Stop an old lookup when the port name changes or the component closes.
  useEffect(() => {
    return () => {
      requestId.current += 1;
      controller.current?.abort();
    };
  }, [port]);

  // Look up the current port name.
  const findCountry = async () => {
    const requestedPort = port.trim();
    if (!requestedPort) {
      setLookupState({
        query: requestedPort,
        loading: false,
        result: null,
        error: "Enter a port name first.",
        requestId: requestId.current,
      });
      return;
    }

    const currentRequestId = ++requestId.current;
    controller.current?.abort();
    controller.current = new AbortController();
    setLookupState({
      query: requestedPort,
      loading: true,
      result: null,
      error: "",
      requestId: currentRequestId,
    });
    try {
      const { data } = await lookupVesselPortCountry(requestedPort, {
        signal: controller.current.signal,
      });
      if (requestId.current !== currentRequestId) return;
      setLookupState((current) => ({
        ...current,
        loading: false,
        result: data,
      }));
      if (data.status === "found") {
        onCountryFound({
          port: data.port,
          port_country: data.country,
          port_country_code: data.country_code,
          port_type: data.port_type,
        });
      }
    } catch (error) {
      if (requestId.current !== currentRequestId || error.code === "ERR_CANCELED") return;
      setLookupState((current) => ({
        ...current,
        loading: false,
        error:
          error.response?.data?.detail ||
          "Could not look up this port country. Check the connection and try again.",
      }));
    }
  };

  // Apply the port and country selected by the user.
  const chooseCandidate = (candidate) => {
    onCountryFound({
      port: candidate.port,
      port_country: candidate.country,
      port_country_code: candidate.country_code,
      port_type: candidate.port_type,
    });
    setLookupState((current) => ({
      ...current,
      result: {
        ...current.result,
        status: "found",
        port: candidate.port,
        country: candidate.country,
        country_code: candidate.country_code,
        port_type: candidate.port_type,
      },
    }));
  };

  const lookup = lookupState.query === port.trim() ? lookupState.result : null;
  const error = lookupState.query === port.trim() ? lookupState.error : "";
  const loading = lookupState.query === port.trim() && lookupState.loading;

  return (
    <div className="vessel-port-country-lookup">
      <button
        type="button"
        className="vessel-port-country-lookup__button"
        onClick={findCountry}
        disabled={loading || !port.trim()}
      >
        {loading ? "LOOKING UP PORT…" : "FIND PORT COUNTRY"}
      </button>
      {country && (
        <p className="vessel-port-country-lookup__result">
          Port country: <strong>{country}</strong>
        </p>
      )}
      {lookup?.status === "ambiguous" && (
        <div className="vessel-port-country-lookup__candidates">
          <p>More than one port matches. Select the correct port:</p>
          {lookup.candidates.map((candidate) => (
            <button
              type="button"
              key={`${candidate.port_code}-${candidate.country_code}`}
              onClick={() => chooseCandidate(candidate)}
            >
              {candidate.port} — {candidate.country}
            </button>
          ))}
        </div>
      )}
      {lookup?.status === "not_found" && (
        <p className="vessel-port-country-lookup__message">
          No matching mapped port found. Check the spelling or enter a more specific port name.
        </p>
      )}
      {error && (
        <p className="vessel-port-country-lookup__message">{error}</p>
      )}
      <small className="vessel-port-country-lookup__attribution">
        Matched against the bundled world port list. Port type identifies the port location,
        not the vessel&apos;s flag or registration country.
      </small>
    </div>
  );
}
