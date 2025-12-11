document.addEventListener("DOMContentLoaded", async () => {
  const canvas = document.getElementById("drawingCanvas");
  const ctx = canvas.getContext("2d");
  const statusBar = document.getElementById("status");
  const btnPredire = document.getElementById("predictButton");
  const resultText = document.getElementById("predictionResult");

  let session;

  console.log("chargement modele");
  try {
    session = await ort.InferenceSession.create("./model.onnx");

    console.log("✅ Modèle chargé avec succès !", session);
    console.log("   - Input names:", session.inputNames);
    console.log("   - Output names:", session.outputNames);

    statusBar.innerText = "Modèle chargé avec succès (ONNX)";
    statusBar.classList.add("show");
  } catch (e) {
    console.error("❌ Échec ❌: ", e);
    statusBar.innerText = "❌Erreur de chargement du modèle !❌";
    statusBar.classList.add("show");
  }

  let isDrawing = false;
  let lastX = 0;
  let lastY = 0;

  ctx.strokeStyle = "#FFFFFF";
  ctx.lineWidth = 18; // Épaisseur de la ligne pour ressembler à un doigt ou un gros pinceau
  ctx.lineCap = "round"; // Bords arrondis pour les lignes
  ctx.lineJoin = "round"; // Coins arrondis

  function draw(e) {
    if (!isDrawing) return; // Arrête la fonction si nous ne sommes pas en train de dessiner

    let rect = canvas.getBoundingClientRect();
    let clientX, clientY;

    if (e.touches && e.touches.length > 0) {
      clientX = e.touches[0].clientX;
      clientY = e.touches[0].clientY;
    } else {
      clientX = e.clientX;
      clientY = e.clientY;
    }

    const x = clientX - rect.left;
    const y = clientY - rect.top;

    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(x, y);
    ctx.stroke();

    [lastX, lastY] = [x, y];
  }

  function startDrawing(e) {
    e.preventDefault(); // Empêche les actions par défaut (comme le défilement sur mobile)
    isDrawing = true;
    // Met à jour les coordonnées initiales
    let rect = canvas.getBoundingClientRect();
    let clientX, clientY;

    if (e.touches && e.touches.length > 0) {
      clientX = e.touches[0].clientX;
      clientY = e.touches[0].clientY;
    } else {
      clientX = e.clientX;
      clientY = e.clientY;
    }

    [lastX, lastY] = [clientX - rect.left, clientY - rect.top];
  }

  // Arrêter le dessin
  function stopDrawing() {
    isDrawing = false;
  }

  // Événements pour la souris
  canvas.addEventListener("mousedown", startDrawing);
  canvas.addEventListener("mousemove", draw);
  canvas.addEventListener("mouseup", stopDrawing);
  canvas.addEventListener("mouseout", stopDrawing); // Arrête si la souris quitte le canvas

  // Événements pour le toucher (écrans tactiles)
  canvas.addEventListener("touchstart", startDrawing);
  canvas.addEventListener("touchmove", draw);
  canvas.addEventListener("touchend", stopDrawing);

  // 4. Fonction Effacer
  const clearButton = document.getElementById("clearButton");
  clearButton.addEventListener("click", () => {
    // Efface tout le rectangle du canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    // Si le fond est noir, le .clearRect() le rend transparent,
    // donc il faut redessiner un fond noir ou utiliser une méthode alternative.
    // Puisque nous avons mis le background-color: #000 en CSS,
    // .clearRect() fonctionne bien pour donner l'impression d'effacer.
  });

  btnPredire.addEventListener("click", async () => {
    if (!session) {
      alert("Model non chargé");
      return;
    }

    const templateCanvas = document.createElement("canvas");

    templateCanvas.width = 28;
    templateCanvas.height = 28;

    const templateCtx = templateCanvas.getContext("2d");

    templateCtx.drawImage(canvas, 0, 0, 28, 28);

    const imgData = templateCtx.getImageData(0, 0, 28, 28);
    const pixels = imgData.data;

    const floatInput = new Float32Array(28 * 28);
    for (let i = 0; i < 28 * 28; i++) {
      // pixels[i * 4] = Canal Rouge (0 à 255)
      // On divise par 255.0 pour avoir entre 0.0 et 1.0
      floatInput[i] = pixels[i * 4] / 255.0;
    }

    // Création du Tensor ONNX
    const inputTensor = new ort.Tensor("float32", floatInput, [1, 1, 28, 28]);
    console.log("D. Tensor ONNX créé :", inputTensor);

    // E. Inférence
    try {
      const inputName = session.inputNames[0];
      const outputName = session.outputNames[0];

      console.log(
        `E. Exécution du modèle (Input: ${inputName} -> Output: ${outputName})...`
      );

      const feeds = {};
      feeds[inputName] = inputTensor;

      const start = performance.now();
      const results = await session.run(feeds);
      const end = performance.now();
      console.log(`✅ Terminée en ${(end - start).toFixed(2)} ms`);

      const outputData = results[outputName].data;

      let maxVal = -Infinity;
      let maxIndex = -1;

      for (let i = 0; i < outputData.length; i++) {
        if (outputData[i] > maxVal) {
          maxVal = outputData[i];
          maxIndex = i;
        }
      }

      console.log(
        `F. Résultat final : Chiffre ${maxIndex} (Confiance/Score : ${maxVal})`
      );
      resultText.innerText = `Chiffre : ${maxIndex} / Probabilité : ${(
        maxVal * 10
      ).toFixed(2)}%`;
    } catch (e) {
      console.error("❌ Erreur pendant la prédiction :", e);
      resultText.innerText = "Erreur...";
    }
  });
});
