import React, { useRef, useEffect, useState } from 'react';
import axios from 'axios';

const VideoStream = () => {
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const [processedFrame, setProcessedFrame] = useState(null);

    useEffect(() => {
        // Access the user's webcam
        navigator.mediaDevices.getUserMedia({ video: true })
            .then((stream) => {
                videoRef.current.srcObject = stream;
                videoRef.current.play();
            })
            .catch((err) => {
                console.error("Error accessing webcam: ", err);
            });
    }, []);

    const captureFrame = () => {
        const canvas = canvasRef.current;
        const context = canvas.getContext("2d");
        context.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
        const frame = canvas.toDataURL("image/png");
        
        axios.post('http://localhost:8000/process_frame', { frame })
            .then(response => {
                setProcessedFrame(`data:image/jpeg;base64,${response.data.frame}`);
            })
            .catch(error => {
                console.error("Error processing frame: ", error);
            });
    };

    return (
        <div>
            <video ref={videoRef} width="640" height="480" />
            <canvas ref={canvasRef} width="640" height="480" style={{ display: 'none' }} />
            <button onClick={captureFrame}>Capture Frame</button>
            {processedFrame && <img src={processedFrame} alt="Processed Frame" />}
        </div>
    );
};

export default VideoStream;