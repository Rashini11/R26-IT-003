import {
  RadarDatabaseHistory,
  SimulationDatabaseHistory,
} from "./DatabaseHistory";


export default function DatabaseViewer() {

  return (
    <div className="database-viewer">

      <div className="page-heading">

        <div>
          <h1>Database</h1>

          <p>
            Stored radar classifications and live-simulation runs when persistence is available.
          </p>
        </div>

      </div>


      <RadarDatabaseHistory />


      <SimulationDatabaseHistory />

    </div>
  );
}
