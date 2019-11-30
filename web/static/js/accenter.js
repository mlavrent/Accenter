URL = window.URL;

let infoTooltips;
let userMediaStream;
let rec;
let input;

let AudioContext = window.AudioContext || window.webkitAudioContext;
let audioContext;

let recordBtn = $(".mic_btn");

recordBtn.click(function() {
    toggleRecording();
});


function toggleRecording() {
	if (rec && rec.recording){
		stopRecording();
	} else {
		startRecording();
	}
}


function startRecording() {
    // Set up recording constraints
    let constraints = { audio: true, video:false };

	navigator.mediaDevices.getUserMedia(constraints).then(function(stream) {
		// Create an audio context
		audioContext = new AudioContext();

		userMediaStream = stream;
		input = audioContext.createMediaStreamSource(stream);

		// Create Recorder object and record 2 channel sound
		rec = new Recorder(input, { numChannels:2 });
		// Start recording
		rec.record();
	}).catch(function(error) {
	});
}

function stopRecording() {
	// Stop recording
	rec.stop();
	// Stop microphone access
	userMediaStream.getAudioTracks()[0].stop();
	// Create WAV blob and send to server
	rec.exportWAV(test);
	rec = null;
}

function test(blob) {
	let url = (window.URL || window.webkitURL).createObjectURL(blob);
	let player = document.getElementById("audioPlayer");
	player.src = url;
	player.load();

	let httpRequest = new XMLHttpRequest();
	httpRequest.onload = function() {
		let detectedText = this.responseText;
		$("#accentPred").text(detectedText);
	};
	httpRequest.open("POST", "/classify", true);
	httpRequest.send(blob);
}


$(document).ready(function () {
    initTooltips();
});


/**
 * Initializes the page tooltips
 */
function initTooltips() {
    infoTooltips = tippy(".info", {
        animation: "scale",
        arrow: true,
        arrowType: "round",
        theme: "accenter",
        hideOnClick: true,
        inertia: true,
        sticky: true,
        placement: "bottom",
    });
}
