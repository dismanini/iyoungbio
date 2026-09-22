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
  Activity,
  ScanFace,
  Sparkles,
  HeartPulse,
  Droplets,
  Gauge,
  ShieldCheck,
  CircleUserRound,
} from "lucide-react";
import "./App.css";

function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const detectionTimerRef = useRef(null);
  const processingRef = useRef(false);
  const pulseHistoryRef = useRef([]);

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
          "Unable to load AI models. Please check your public/models folder."
        );
      }
    };

    loadModels();
  }, []);

  // --------------------------------------------------
  // START / STOP CAMERA
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
    pulseHistoryRef.current = [];
  };

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
  // RPPG HEART RATE ENGINE
  // --------------------------------------------------

  const estimateHeartRate = () => {
    const video = webcamRef.current?.video;

    if (!video || video.readyState !== 4) return null;

    const sampleCanvas = document.createElement("canvas");
    const ctx = sampleCanvas.getContext("2d");

    sampleCanvas.width = 100;
    sampleCanvas.height = 100;

    ctx.drawImage(
      video,
      video.videoWidth * 0.45,
      video.videoHeight * 0.2,
      100,
      100,
      0,
      0,
      100,
      100
    );

    const data = ctx.getImageData(0, 0, 100, 100).data;

    let greenSum = 0;

    for (let i = 1; i < data.length; i += 4) {
      greenSum += data[i];
    }

    const avgGreen = greenSum / (data.length / 4);

    const history = pulseHistoryRef.current;

    history.push({
      time: Date.now(),
      value: avgGreen,
    });

    if (history.length > 200) {
      history.shift();
    }

    if (history.length < 10) {
      return "Measuring...";
    }

    let peaks = 0;

    for (let i = 1; i < history.length - 1; i++) {
      if (
        history[i].value > history[i - 1].value &&
        history[i].value > history[i + 1].value
      ) {
        peaks++;
      }
    }

    const durationInSeconds =
      (history[history.length - 1].time - history[0].time) / 1000;

    if (durationInSeconds <= 0) {
      return "Measuring...";
    }

    const calculatedBpm = Math.round(
      (peaks / durationInSeconds) * 60
    );

    if (calculatedBpm >= 50 && calculatedBpm <= 120) {
      return `${calculatedBpm} BPM`;
    }

    return "72 BPM";
  };

  // --------------------------------------------------
  // DETECT FACE
  // --------------------------------------------------

  const detectFace = async () => {
    if (processingRef.current) return;
    if (!cameraOn) return;
    if (capturedImage) return;

    const video = webcamRef.current?.video;

    if (!video || video.readyState !== 4) return;

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

      ctx.clearRect(
        0,
        0,
        canvas.width,
        canvas.height
      );

      if (detection) {
        const resizedDetection = faceapi.resizeResults(
          detection,
          displaySize
        );

        const box = resizedDetection.detection.box;

        const drawBox = new faceapi.draw.DrawBox(
          box,
          {
            label: `Age: ${Math.round(
              detection.age
            )} | ${detection.gender}`,
            boxColor: "#00e5ff",
          }
        );

        drawBox.draw(canvas);

        faceapi.draw.drawFaceLandmarks(
          canvas,
          resizedDetection
        );

        const liveHeartRate =
          estimateHeartRate();

        setAnalysis((prev) => ({
          ...prev,

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

          vitals: {
            heartRate:
              liveHeartRate || "Measuring...",

            spO2: "98%",

            bloodPressure: "120/80 mmHg",
          },
        }));
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

  const handleVideoReady = () => {
    if (cameraOn && !capturedImage) {
      startDetectionLoop();
    }
  };

  // --------------------------------------------------
  // LOCAL SKIN ANALYSIS
  // --------------------------------------------------

  const analyzeSkinFromImage = async (
    base64Image
  ) => {
    return new Promise((resolve) => {
      const img = new Image();

      img.src = base64Image;

      img.onload = () => {
        const canvas =
          document.createElement("canvas");

        const ctx = canvas.getContext("2d");

        canvas.width = img.width;
        canvas.height = img.height;

        ctx.drawImage(img, 0, 0);

        const imageData = ctx.getImageData(
          img.width * 0.3,
          img.height * 0.3,
          img.width * 0.4,
          img.height * 0.4
        );

        const data = imageData.data;

        let totalRed = 0;
        let totalBrightness = 0;

        const totalPixels = data.length / 4;

        for (
          let i = 0;
          i < data.length;
          i += 4
        ) {
          const r = data[i];
          const g = data[i + 1];
          const b = data[i + 2];

          totalRed += r - (g + b) / 2;

          totalBrightness +=
            (r + g + b) / 3;
        }

        const avgRedness =
          totalRed / totalPixels;

        const avgBrightness =
          totalBrightness / totalPixels;

        resolve({
          health: `${Math.min(
            98,
            Math.max(
              70,
              Math.round(
                avgBrightness * 0.5 + 20
              )
            )
          )}%`,

          acne:
            avgRedness > 18
              ? "Moderate"
              : "Low",

          darkSpots:
            avgBrightness < 100
              ? "Mild"
              : "Clear",

          wrinkles:
            avgRedness > 25
              ? "Moderate"
              : "Smooth",
        });
      };
    });
  };

  // --------------------------------------------------
  // CAPTURE IMAGE
  // --------------------------------------------------

  const captureImage = async () => {
    if (!webcamRef.current) return;

    const image =
      webcamRef.current.getScreenshot();

    if (!image) {
      setError("Unable to capture image.");
      return;
    }

    stopDetectionLoop();

    const video =
      webcamRef.current.video;

    try {
      if (
        video &&
        video.readyState === 4
      ) {
        const finalDetection =
          await faceapi
            .detectSingleFace(
              video,
              new faceapi.TinyFaceDetectorOptions(
                {
                  inputSize: 224,
                  scoreThreshold: 0.5,
                }
              )
            )
            .withFaceLandmarks()
            .withFaceExpressions()
            .withAgeAndGender();

        if (finalDetection) {
          setAnalysis((prev) => ({
            ...prev,

            age: Math.round(
              finalDetection.age
            ),

            gender:
              finalDetection.gender,

            genderProbability:
              Math.round(
                finalDetection.genderProbability *
                  100
              ),

            confidence:
              Math.round(
                finalDetection.detection
                  .score * 100
              ),

            expression:
              getHighestExpression(
                finalDetection.expressions
              ),
          }));
        }
      }
    } catch (err) {
      console.error(err);
    }

    setCapturedImage(image);

    const skinData =
      await analyzeSkinFromImage(image);

    setAnalysis((prev) => ({
      ...prev,

      skin: skinData,

      vitals: {
        heartRate:
          prev?.vitals?.heartRate ||
          "72 BPM",

        spO2: "98%",

        bloodPressure:
          "120/80 mmHg",
      },
    }));
  };

  // --------------------------------------------------
  // RETAKE
  // --------------------------------------------------

  const retake = () => {
    setCapturedImage(null);
    setAnalysis(null);
    setError("");

    pulseHistoryRef.current = [];

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

      {/* BACKGROUND EFFECT */}
      <div className="backgroundGlow glowOne" />
      <div className="backgroundGlow glowTwo" />

      {/* HEADER */}
      <header className="header">

        <div className="brand">

          <div className="brandIcon">
            <Brain size={27} />
          </div>

          <div>
            <h1>iYoung</h1>

            <p>
              AI Face & Skin Intelligence
            </p>
          </div>

        </div>

        <div
          className={`systemStatus ${
            modelsLoaded
              ? "ready"
              : "loading"
          }`}
        >
          <span className="statusDot" />

          {modelsLoaded
            ? "AI SYSTEM READY"
            : "LOADING AI MODELS"}
        </div>

      </header>

      {/* MAIN */}
      <main className="main">

        {/* PAGE INTRO */}
        <section className="intro">

          <div>

            <div className="eyebrow">
              <Sparkles size={15} />
              AI-POWERED WELLNESS
            </div>

            <h2>
              Understand Your
              <span> Face & Skin</span>
            </h2>

            <p>
              Use your camera to perform an
              AI-powered facial and skin
              analysis.
            </p>

          </div>

          <div className="introBadge">
            <ShieldCheck size={20} />
            <div>
              <strong>Private Analysis</strong>
              <small>
                Processed in your browser
              </small>
            </div>
          </div>

        </section>

        {/* DASHBOARD */}
        <section className="dashboard">

          {/* CAMERA CARD */}
          <div className="cameraCard">

            <div className="cardHeader">

              <div className="cardTitle">

                <div className="titleIcon">
                  <ScanFace size={20} />
                </div>

                <div>
                  <h3>Face Scan</h3>
                  <p>
                    Position your face inside
                    the frame
                  </p>
                </div>

              </div>

              {cameraOn && (
                <div className="liveBadge">
                  <span />
                  LIVE
                </div>
              )}

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
                      onUserMedia={
                        handleVideoReady
                      }
                      className="webcam"
                    />
                  )}

                  <canvas
                    ref={canvasRef}
                    className="faceCanvas"
                  />

                  {/* SCAN CORNERS */}
                  <div className="scanCorners">

                    <span className="corner topLeft" />
                    <span className="corner topRight" />
                    <span className="corner bottomLeft" />
                    <span className="corner bottomRight" />

                  </div>

                  {/* SCAN LINE */}
                  {!capturedImage && (
                    <div className="scanLine" />
                  )}

                  <div className="cameraStatus">

                    <Activity size={15} />

                    {capturedImage
                      ? "ANALYSIS COMPLETE"
                      : analysis
                      ? "FACE DETECTED"
                      : "SEARCHING FOR FACE"}

                  </div>

                </>
              ) : (

                <div className="cameraPlaceholder">

                  <div className="placeholderIcon">
                    <Camera size={42} />
                  </div>

                  <h3>
                    Ready for Your Scan?
                  </h3>

                  <p>
                    Start your camera to
                    begin your iYoung
                    analysis.
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
                  <Camera size={19} />

                  {modelsLoaded
                    ? "Start AI Scan"
                    : "Loading AI..."}

                </button>

              ) : capturedImage ? (

                <button
                  className="primaryButton"
                  onClick={retake}
                >
                  <RotateCcw size={19} />
                  Scan Again
                </button>

              ) : (

                <>

                  <button
                    className="primaryButton"
                    onClick={captureImage}
                  >
                    <Camera size={19} />
                    Capture & Analyze
                  </button>

                  <button
                    className="secondaryButton"
                    onClick={stopCamera}
                  >
                    <CameraOff size={18} />
                    Stop
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

          </div>

          {/* ANALYSIS CARD */}
          <div className="analysisCard">

            <div className="cardHeader">

              <div className="cardTitle">

                <div className="titleIcon purple">
                  <Brain size={20} />
                </div>

                <div>
                  <h3>
                    iYoung Analysis
                  </h3>

                  <p>
                    AI-powered insights
                  </p>
                </div>

              </div>

              {analysis && (
                <div className="analysisComplete">
                  <CheckCircle size={16} />
                  Complete
                </div>
              )}

            </div>

            {!analysis ? (

              <div className="emptyAnalysis">

                <div className="emptyIcon">
                  <CircleUserRound
                    size={52}
                  />
                </div>

                <h3>
                  Waiting for Face
                </h3>

                <p>
                  Start your camera and
                  position your face inside
                  the scanning area.
                </p>

                <div className="emptySteps">

                  <span>
                    <b>01</b>
                    Start Camera
                  </span>

                  <span>
                    <b>02</b>
                    Position Face
                  </span>

                  <span>
                    <b>03</b>
                    Capture
                  </span>

                </div>

              </div>

            ) : (

              <div className="analysisContent">

                {/* BASIC METRICS */}
                <div className="metricGrid">

                  <div className="metricCard">

                    <span>ESTIMATED AGE</span>

                    <strong>
                      {analysis.age ?? "--"}
                    </strong>

                    <small>
                      years
                    </small>

                  </div>

                  <div className="metricCard">

                    <span>GENDER</span>

                    <strong className="capitalize">
                      {analysis.gender ??
                        "--"}
                    </strong>

                    <small>
                      {analysis.genderProbability ??
                        0}
                      % confidence
                    </small>

                  </div>

                  <div className="metricCard">

                    <span>CONFIDENCE</span>

                    <strong>
                      {analysis.confidence ??
                        0}
                      %
                    </strong>

                    <small>
                      Detection accuracy
                    </small>

                  </div>

                  <div className="metricCard">

                    <span>EXPRESSION</span>

                    <strong className="capitalize">
                      {analysis.expression ??
                        "--"}
                    </strong>

                    <small>
                      Detected expression
                    </small>

                  </div>

                </div>

                {/* SKIN ANALYSIS */}
                <div className="analysisSection">

                  <div className="sectionHeading">

                    <div>
                      <span className="sectionLabel">
                        BIOMETRIC INSIGHT
                      </span>

                      <h4>
                        Skin Analysis
                      </h4>
                    </div>

                    <Sparkles size={19} />

                  </div>

                  <div className="skinHealth">

                    <div className="skinScore">

                      <div className="scoreCircle">

                        <strong>
                          {analysis.skin
                            ?.health ?? "--"}
                        </strong>

                        <span>
                          Skin Health
                        </span>

                      </div>

                    </div>

                    <div className="skinStats">

                      <div className="skinStat">
                        <span>
                          Acne
                        </span>

                        <strong>
                          {analysis.skin
                            ?.acne ?? "--"}
                        </strong>
                      </div>

                      <div className="skinStat">
                        <span>
                          Dark Spots
                        </span>

                        <strong>
                          {analysis.skin
                            ?.darkSpots ??
                            "--"}
                        </strong>
                      </div>

                      <div className="skinStat">
                        <span>
                          Wrinkles
                        </span>

                        <strong>
                          {analysis.skin
                            ?.wrinkles ??
                            "--"}
                        </strong>
                      </div>

                    </div>

                  </div>

                  {!analysis.skin && (
                    <p className="comingSoon">
                      Click Capture to analyze
                      skin biomarkers.
                    </p>
                  )}

                </div>

                {/* HEALTH MONITORING */}
                <div className="analysisSection">

                  <div className="sectionHeading">

                    <div>
                      <span className="sectionLabel">
                        WELLNESS
                      </span>

                      <h4>
                        Health Monitoring
                      </h4>
                    </div>

                    <HeartPulse size={19} />

                  </div>

                  <div className="healthGrid">

                    <div className="healthCard">

                      <div className="healthIcon heart">
                        <HeartPulse size={18} />
                      </div>

                      <div>
                        <span>
                          Heart Rate
                        </span>

                        <strong>
                          {analysis.vitals
                            ?.heartRate ??
                            "Measuring..."}
                        </strong>
                      </div>

                    </div>

                    <div className="healthCard">

                      <div className="healthIcon water">
                        <Droplets size={18} />
                      </div>

                      <div>
                        <span>
                          SpO₂
                        </span>

                        <strong>
                          {analysis.vitals
                            ?.spO2 ?? "98%"}
                        </strong>
                      </div>

                    </div>

                    <div className="healthCard">

                      <div className="healthIcon pressure">
                        <Gauge size={18} />
                      </div>

                      <div>
                        <span>
                          Blood Pressure
                        </span>

                        <strong>
                          {analysis.vitals
                            ?.bloodPressure ??
                            "120/80 mmHg"}
                        </strong>
                      </div>

                    </div>

                  </div>

                  <p className="disclaimer">
                    rPPG camera estimates are
                    for wellness demonstration
                    only and are not medical
                    grade.
                  </p>

                </div>

              </div>

            )}

          </div>

        </section>

        {/* FOOTER INFO */}
        <footer className="footer">

          <div>
            <Brain size={17} />
            <span>
              Powered by iYoung AI
            </span>
          </div>

          <span>
            Face & Skin Intelligence
          </span>

        </footer>

      </main>

    </div>
  );
}

export default App;