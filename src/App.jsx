import { useEffect, useRef, useState } from "react";
import Webcam from "react-webcam";
import * as faceapi from "face-api.js";
import {
  Camera,
  CameraOff,
  RotateCcw,
  User,
  Brain,
  CheckCircle,
  AlertCircle,
} from "lucide-react";
import "./App.css";

function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const detectionTimerRef = useRef(null);
  const processingRef = useRef(false);

  const [modelsLoaded, setModelsLoaded] = useState(false);
  const [cameraOn, setCameraOn] = useState(false);
  const [capturedImage, setCapturedImage] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [error, setError] = useState("");

  // --------------------------------------------------
  // LOAD FACE-API MODELS
  // --------------------------------------------------

  useEffect(() => {
    const loadModels = async () => {
      try {
        setError("");

        const MODEL_URL = "/models";

        await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL);
        await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);
        await faceapi.nets.faceExpressionNet.loadFromUri(MODEL_URL);
        await faceapi.nets.ageGenderNet.loadFromUri(MODEL_URL);

        setModelsLoaded(true);

        console.log("All face-api.js models loaded successfully.");
      } catch (err) {
        console.error("Model loading error:", err);
        setError(
          "Unable to load AI models. Please check public/models folder."
        );
      }
    };

    loadModels();
  }, []);

  // --------------------------------------------------
  // START CAMERA
  // --------------------------------------------------

  const startCamera = () => {
    if (!modelsLoaded) {
      setError("Please wait until AI models are loaded.");
      return;
    }

    setError("");
    setCameraOn(true);
    setCapturedImage(null);
    setAnalysis(null);
  };

  // --------------------------------------------------
  // STOP CAMERA
  // --------------------------------------------------

  const stopCamera = () => {
    setCameraOn(false);
    stopDetectionLoop();
  };

  // --------------------------------------------------
  // DETECTION LOOP
  // --------------------------------------------------

  const startDetectionLoop = () => {
    stopDetectionLoop();

    detectionTimerRef.current = setInterval(async () => {
      await detectFace();
    }, 500);
  };

  const stopDetectionLoop = () => {
    if (detectionTimerRef.current) {
      clearInterval(detectionTimerRef.current);
      detectionTimerRef.current = null;
    }
  };

  // --------------------------------------------------
  // DETECT FACE
  // --------------------------------------------------

  const detectFace = async () => {
    if (processingRef.current) return;
    if (!cameraOn) return;
    if (capturedImage) return;

    const video = webcamRef.current?.video;

    if (!video) return;

    if (video.readyState !== 4) return;

    processingRef.current = true;

    try {
      const detection = await faceapi
        .detectSingleFace(
          video,
          new faceapi.TinyFaceDetectorOptions({
            inputSize: 224,
            scoreThreshold: 0.5,
          })
        )
        .withFaceLandmarks()
        .withFaceExpressions()
        .withAgeAndGender();

      const canvas = canvasRef.current;

      if (!canvas) {
        processingRef.current = false;
        return;
      }

      const displaySize = {
        width: video.videoWidth,
        height: video.videoHeight,
      };

      faceapi.matchDimensions(canvas, displaySize);

      const ctx = canvas.getContext("2d");

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      if (detection) {
        const resizedDetection = faceapi.resizeResults(
          detection,
          displaySize
        );

        // Draw face box
        const box = resizedDetection.detection.box;

        const drawBox = new faceapi.draw.DrawBox(box, {
          label: `Age: ${Math.round(detection.age)} | ${detection.gender}`,
          boxColor: "#00e5ff",
        });

        drawBox.draw(canvas);

        // Draw landmarks
        faceapi.draw.drawFaceLandmarks(
          canvas,
          resizedDetection
        );

        setAnalysis({
          age: Math.round(detection.age),
          gender: detection.gender,
          genderProbability: Math.round(
            detection.genderProbability * 100
          ),
          confidence: Math.round(
            detection.detection.score * 100
          ),
          expression: getHighestExpression(
            detection.expressions
          ),
        });
      } else {
        setAnalysis(null);
      }
    } catch (err) {
      console.error("Face detection error:", err);
    }

    processingRef.current = false;
  };

  // --------------------------------------------------
  // HIGHEST EXPRESSION
  // --------------------------------------------------

  const getHighestExpression = (expressions) => {
    if (!expressions) return "Unknown";

    return Object.entries(expressions).sort(
      (a, b) => b[1] - a[1]
    )[0][0];
  };

  // --------------------------------------------------
  // CAMERA READY
  // --------------------------------------------------

  const handleVideoReady = () => {
    if (cameraOn && !capturedImage) {
      startDetectionLoop();
    }
  };

  // --------------------------------------------------
  // CAPTURE
  // --------------------------------------------------

  const captureImage = async () => {
    if (!webcamRef.current) return;

    const image = webcamRef.current.getScreenshot();

    if (!image) {
      setError("Unable to capture image.");
      return;
    }

    // IMPORTANT:
    // Stop live detection before freezing the result.
    stopDetectionLoop();

    // Take one final analysis from the current frame.
    const video = webcamRef.current.video;

    try {
      if (video && video.readyState === 4) {
        const finalDetection = await faceapi
          .detectSingleFace(
            video,
            new faceapi.TinyFaceDetectorOptions({
              inputSize: 224,
              scoreThreshold: 0.5,
            })
          )
          .withFaceLandmarks()
          .withFaceExpressions()
          .withAgeAndGender();

        if (finalDetection) {
          setAnalysis({
            age: Math.round(finalDetection.age),
            gender: finalDetection.gender,
            genderProbability: Math.round(
              finalDetection.genderProbability * 100
            ),
            confidence: Math.round(
              finalDetection.detection.score * 100
            ),
            expression: getHighestExpression(
              finalDetection.expressions
            ),
          });
        }
      }
    } catch (err) {
      console.error(err);
    }

    setCapturedImage(image);
  };

  // --------------------------------------------------
  // RETAKE
  // --------------------------------------------------

  const retake = () => {
    setCapturedImage(null);
    setAnalysis(null);
    setError("");

    if (cameraOn) {
      setTimeout(() => {
        startDetectionLoop();
      }, 500);
    }
  };

  // --------------------------------------------------
  // CLEANUP
  // --------------------------------------------------

  useEffect(() => {
    return () => {
      stopDetectionLoop();
    };
  }, []);

  return (
    <div className="app">

      {/* HEADER */}

      <header className="header">
        <div className="logoArea">
          <div className="logoIcon">
            <Brain size={28} />
          </div>

          <div>
            <h1>iYoung</h1>
            <p>AI Face & Skin Analysis</p>
          </div>
        </div>

        <div className="modelStatus">
          {modelsLoaded ? (
            <>
              <CheckCircle size={18} />
              AI Models Ready
            </>
          ) : (
            <>
              <Brain size={18} />
              Loading AI Models...
            </>
          )}
        </div>
      </header>

      {/* MAIN */}

      <main className="main">

        {/* CAMERA SECTION */}

        <section className="cameraCard">

          <div className="sectionTitle">
            <Camera size={22} />
            <h2>Face Scan</h2>
          </div>

          <div className="cameraContainer">

            {cameraOn ? (
              <>
                {capturedImage ? (
                  <img
                    src={capturedImage}
                    alt="Captured face"
                    className="capturedImage"
                  />
                ) : (
                  <Webcam
                    ref={webcamRef}
                    audio={false}
                    screenshotFormat="image/jpeg"
                    videoConstraints={{
                      width: 1280,
                      height: 720,
                      facingMode: "user",
                    }}
                    onUserMedia={handleVideoReady}
                    className="webcam"
                  />
                )}

                <canvas
                  ref={canvasRef}
                  className="faceCanvas"
                />
              </>
            ) : (
              <div className="cameraPlaceholder">
                <CameraOff size={60} />
                <h3>Camera is Off</h3>
                <p>
                  Start the camera to begin your iYoung analysis.
                </p>
              </div>
            )}

          </div>

          {/* CONTROLS */}

          <div className="controls">

            {!cameraOn ? (
              <button
                className="primaryButton"
                onClick={startCamera}
                disabled={!modelsLoaded}
              >
                <Camera size={20} />
                Start Camera
              </button>
            ) : capturedImage ? (
              <button
                className="primaryButton"
                onClick={retake}
              >
                <RotateCcw size={20} />
                Retake
              </button>
            ) : (
              <>
                <button
                  className="primaryButton"
                  onClick={captureImage}
                >
                  <Camera size={20} />
                  Capture
                </button>

                <button
                  className="secondaryButton"
                  onClick={stopCamera}
                >
                  <CameraOff size={20} />
                  Stop Camera
                </button>
              </>
            )}

          </div>

          {error && (
            <div className="error">
              <AlertCircle size={18} />
              {error}
            </div>
          )}

        </section>

        {/* ANALYSIS */}

        <section className="analysisCard">

          <div className="sectionTitle">
            <User size={22} />
            <h2>iYoung Analysis</h2>
          </div>

          {!analysis ? (
            <div className="emptyAnalysis">
              <Brain size={48} />
              <h3>Waiting for Face</h3>
              <p>
                Position your face inside the camera.
              </p>
            </div>
          ) : (
            <>
              <div className="resultGrid">

                <div className="resultBox">
                  <span>Estimated Age</span>
                  <strong>{analysis.age}</strong>
                  <small>years</small>
                </div>

                <div className="resultBox">
                  <span>Gender</span>
                  <strong>{analysis.gender}</strong>
                  <small>
                    {analysis.genderProbability}% confidence
                  </small>
                </div>

                <div className="resultBox">
                  <span>Face Confidence</span>
                  <strong>{analysis.confidence}%</strong>
                  <small>Detection accuracy</small>
                </div>

                <div className="resultBox">
                  <span>Expression</span>
                  <strong>{analysis.expression}</strong>
                  <small>Detected expression</small>
                </div>

              </div>

              {/* SKIN ANALYSIS */}

              <div className="subSection">
                <h3>Skin Analysis</h3>

                <div className="skinGrid">

                  <div>
                    <span>Skin Health</span>
                    <strong>--</strong>
                  </div>

                  <div>
                    <span>Acne</span>
                    <strong>--</strong>
                  </div>

                  <div>
                    <span>Dark Spots</span>
                    <strong>--</strong>
                  </div>

                  <div>
                    <span>Wrinkles</span>
                    <strong>--</strong>
                  </div>

                </div>

                <p className="comingSoon">
                  Skin analysis model will be connected in the
                  next stage.
                </p>
              </div>

              {/* HEALTH */}

              <div className="subSection">

                <h3>Health Monitoring</h3>

                <div className="healthItem">
                  <span>❤️ Heart Rate</span>
                  <strong>Not measured</strong>
                </div>

                <div className="healthItem">
                  <span>💧 SpO₂</span>
                  <strong>Not measured</strong>
                </div>

                <div className="healthItem">
                  <span>🩺 Blood Pressure</span>
                  <strong>Not measured</strong>
                </div>

                <p className="disclaimer">
                  Vital-sign measurements require a validated
                  algorithm or physical sensor.
                </p>

              </div>

            </>
          )}

        </section>

      </main>

    </div>
  );
}

export default App;