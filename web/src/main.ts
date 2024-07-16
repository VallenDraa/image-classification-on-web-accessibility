import './style.css';

import * as tf from '@tensorflow/tfjs';

let classes: string[] | null = null;

async function loadModel(): Promise<tf.GraphModel> {
	const model = await tf.loadGraphModel('/image_classifier_model/model.json');
	return model;
}

function preprocessImage(image: HTMLImageElement): tf.Tensor {
	return tf.tidy(() => {
		const tensor = tf.browser
			.fromPixels(image)
			.resizeNearestNeighbor([32, 32])
			.toFloat()
			.div(tf.scalar(255.0))
			.expandDims();

		return tensor;
	});
}

async function generateAltAttr(image: HTMLImageElement): Promise<void> {
	if (!classes) {
		const response = await fetch('/image_classifier_model/classes.json');
		classes = await response.json();
	}

	const model = await loadModel();
	const processedImage = preprocessImage(image);
	const prediction = model.predict(processedImage);

	if (prediction instanceof tf.Tensor) {
		const predictedClass = (await prediction.argMax(-1).data())[0];
		image.alt = `Gambar ${classes![predictedClass]}`;
	} else {
		alert('Gagal melakukan prediksi gambar');
	}
}

// Gambar
const catPic = document.getElementById('picture-1') as HTMLImageElement;
const unknownPic = document.getElementById('picture-2') as HTMLImageElement;

// Tombol prediksi
const generateAltBtn = document.getElementById(
	'generate-alt',
) as HTMLButtonElement;

// Status alt attribute
const statusAltAttribute = document.querySelector(
	'.status',
) as HTMLParagraphElement;

// Event listener
generateAltBtn.onclick = async () => {
	await generateAltAttr(catPic);
	await generateAltAttr(unknownPic);

	statusAltAttribute.innerHTML =
		'<code>alt</code> attribute berhasil digenerate';
};
