import AVFAudio
import Foundation
import Speech

/// Wraps `SFSpeechRecognizer` + `AVAudioEngine` for live dictation.
///
/// Uses on-device recognition when the system supports it (keeps sensitive
/// expense descriptions off Apple's servers).
@MainActor
final class SpeechTranscriber: ObservableObject {
    enum TranscriberError: LocalizedError {
        case speechNotAuthorized
        case microphoneNotAuthorized
        case recognizerUnavailable

        var errorDescription: String? {
            switch self {
            case .speechNotAuthorized:
                return "Speech recognition permission denied. Enable it in Settings → ExpenseParser."
            case .microphoneNotAuthorized:
                return "Microphone permission denied. Enable it in Settings → ExpenseParser."
            case .recognizerUnavailable:
                return "Speech recognizer unavailable for this locale."
            }
        }
    }

    @Published private(set) var transcript: String = ""
    @Published private(set) var isRecording: Bool = false
    @Published var errorMessage: String?

    private let recognizer: SFSpeechRecognizer?
    private let audioEngine = AVAudioEngine()
    private var request: SFSpeechAudioBufferRecognitionRequest?
    private var task: SFSpeechRecognitionTask?

    init(locale: Locale = .current) {
        self.recognizer = SFSpeechRecognizer(locale: locale)
    }

    func toggle() async {
        if isRecording {
            stop()
        } else {
            await start()
        }
    }

    func start() async {
        guard !isRecording else { return }
        transcript = ""
        errorMessage = nil
        do {
            try await requestPermissions()
            try beginRecording()
            isRecording = true
        } catch {
            errorMessage = error.localizedDescription
            cleanup()
        }
    }

    func stop() {
        guard isRecording else { return }
        cleanup()
        isRecording = false
    }

    private func cleanup() {
        if audioEngine.isRunning {
            audioEngine.stop()
            audioEngine.inputNode.removeTap(onBus: 0)
        }
        request?.endAudio()
        task?.finish()
        task = nil
        request = nil
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
    }

    private func requestPermissions() async throws {
        let speechStatus: SFSpeechRecognizerAuthorizationStatus = await withCheckedContinuation { c in
            SFSpeechRecognizer.requestAuthorization { c.resume(returning: $0) }
        }
        guard speechStatus == .authorized else { throw TranscriberError.speechNotAuthorized }

        let micGranted: Bool = await withCheckedContinuation { c in
            AVAudioApplication.requestRecordPermission { c.resume(returning: $0) }
        }
        guard micGranted else { throw TranscriberError.microphoneNotAuthorized }
    }

    private func beginRecording() throws {
        guard let recognizer, recognizer.isAvailable else {
            throw TranscriberError.recognizerUnavailable
        }

        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.record, mode: .measurement, options: .duckOthers)
        try session.setActive(true, options: .notifyOthersOnDeactivation)

        let request = SFSpeechAudioBufferRecognitionRequest()
        request.shouldReportPartialResults = true
        request.requiresOnDeviceRecognition = recognizer.supportsOnDeviceRecognition
        self.request = request

        let inputNode = audioEngine.inputNode
        let format = inputNode.outputFormat(forBus: 0)
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: format) { buffer, _ in
            request.append(buffer)
        }

        task = recognizer.recognitionTask(with: request) { [weak self] result, error in
            Task { @MainActor in
                guard let self else { return }
                if let result {
                    self.transcript = result.bestTranscription.formattedString
                }
                if error != nil || result?.isFinal == true {
                    self.stop()
                }
            }
        }

        audioEngine.prepare()
        try audioEngine.start()
    }
}
