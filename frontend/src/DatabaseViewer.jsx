import {
  RadarDatabaseHistory,
  SimulationDatabaseHistory,
} from "./DatabaseHistory";


export default function DatabaseViewer() {

  return (
    <div className="database-viewer">

      <div className="page-heading">

        <div>
          <h1>
            OceanIQ Database
          </h1>

          <p>
            Stored Radar classification and
            Live Simulation records.
          </p>
        </div>

      </div>


      <RadarDatabaseHistory />


      <SimulationDatabaseHistory />

    </div>
  );
}
