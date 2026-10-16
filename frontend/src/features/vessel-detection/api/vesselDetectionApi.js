import axios from "axios";

const API_BASE_URL = "http://localhost:8000";
const VESSEL_DETECTION_URL = `${API_BASE_URL}/predict-boat-detection`;
const VESSEL_VIDEO_URL = `${API_BASE_URL}/predict-boat-video`;
const VESSEL_RECORDS_URL = `${API_BASE_URL}/vessel-detection-records`;
const VESSEL_HISTORY_URL = `${API_BASE_URL}/vessel-detection-history`;
const VESSEL_PORT_COUNTRY_URL = `${API_BASE_URL}/vessel-detection/port-country`;

// Send vessel images or videos for analysis.
export const predictVesselImage = (form) =>
  axios.post(VESSEL_DETECTION_URL, form, {
    headers: { "Content-Type": "multipart/form-data" },
  });

export const predictVesselVideo = (form) =>
  axios.post(VESSEL_VIDEO_URL, form, {
    headers: { "Content-Type": "multipart/form-data" },
  });

// Load saved vessel records.
export const getVesselHistory = () => axios.get(VESSEL_HISTORY_URL);

// Find a port and its country.
export const lookupVesselPortCountry = (port, options = {}) =>
  axios.get(VESSEL_PORT_COUNTRY_URL, {
    params: { port },
    ...options,
  });

// Save a vessel record with its uploaded images.
export const createVesselRecord = (form) =>
  axios.post(VESSEL_RECORDS_URL, form, {
    headers: { "Content-Type": "multipart/form-data" },
  });

// Update or delete a saved vessel record.
export const updateVesselRecord = (recordId, updates) =>
  axios.put(`${VESSEL_HISTORY_URL}/${recordId}`, updates);

export const deleteVesselRecord = (recordId) =>
  axios.delete(`${VESSEL_HISTORY_URL}/${recordId}`);
