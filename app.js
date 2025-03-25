
import React, { useState } from "react";
import axios from "axios";

function App() {
     const [attackData, setAttackData] = useState(null);

     const checkSecurity = async () => {
          const sampleData = [0.5, 0.7, 0.2, 0.9, 0.1]; // Example network traffic data
          const response = await axios.post("http://localhost:5000/detect", sampleData);
          setAttackData(response.data);
     };

     return (
          <div className="container">
               <h2>Cyber Attack Detection in WSN</h2>
               <button onClick={checkSecurity}>Run Security Check</button>

               {attackData && (
                    <div>
                         <h3>Detection Results</h3>
                         <pre>{JSON.stringify(attackData, null, 2)}</pre>
                    </div>
               )}
          </div>
     );
}

export default App;
