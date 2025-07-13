/**
 * ARES Edge System - LIDAR and Audio Integration
 * Copyright (c) 2024 DELFICTUS I/O LLC
 * 
 * Full implementation of LIDAR and audio processing with neuromorphic integration
 */

#include "neuromorphic_core.h"
#include "custom_neuron_models.cpp"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/vlp_grabber.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/features/normal_3d.h>
#include <portaudio.h>
#include <sndfile.h>
#include <Eigen/Dense>

namespace ares {
namespace neuromorphic {

/**
 * LIDAR point cloud processor with neuromorphic object detection
 */
class LIDARNeuromorphicProcessor {
private:
    // LIDAR components
    std::unique_ptr<pcl::VLPGrabber> velodyne_grabber;
    pcl::PointCloud<pcl::PointXYZI>::Ptr current_cloud;
    
    // Neuromorphic components
    std::unique_ptr<GridCellNeuron> spatial_neurons;      // Spatial mapping
    std::unique_ptr<PatternNeuron> object_neurons;        // Object recognition
    std::unique_ptr<BistableNeuron> decision_neurons;     // Threat/no-threat
    
    // Processing pipeline
    std::thread lidar_thread;
    std::atomic<bool> running{false};
    
    // Object detection parameters
    struct DetectedObject {
        Eigen::Vector3f position;
        Eigen::Vector3f velocity;
        float confidence;
        int object_class;
        std::vector<float> neuromorphic_signature;
    };
    
    std::vector<DetectedObject> detected_objects;
    std::mutex objects_mutex;
    
    // Voxel grid for downsampling
    pcl::VoxelGrid<pcl::PointXYZI> voxel_filter;
    
    // Kalman filter for tracking
    class ObjectTracker {
    private:
        Eigen::Matrix<float, 6, 6> F;  // State transition
        Eigen::Matrix<float, 3, 6> H;  // Measurement matrix
        Eigen::Matrix<float, 6, 6> Q;  // Process noise
        Eigen::Matrix<float, 3, 3> R;  // Measurement noise
        Eigen::Matrix<float, 6, 6> P;  // Error covariance
        Eigen::Matrix<float, 6, 1> x;  // State [x, y, z, vx, vy, vz]
        
    public:
        ObjectTracker() {
            // Initialize matrices
            F = Eigen::Matrix<float, 6, 6>::Identity();
            float dt = 0.1f;  // 10 Hz update rate
            F(0, 3) = dt; F(1, 4) = dt; F(2, 5) = dt;
            
            H = Eigen::Matrix<float, 3, 6>::Zero();
            H(0, 0) = 1; H(1, 1) = 1; H(2, 2) = 1;
            
            Q = Eigen::Matrix<float, 6, 6>::Identity() * 0.1f;
            R = Eigen::Matrix<float, 3, 3>::Identity() * 0.5f;
            P = Eigen::Matrix<float, 6, 6>::Identity() * 100.0f;
            x = Eigen::Matrix<float, 6, 1>::Zero();
        }
        
        void predict() {
            x = F * x;
            P = F * P * F.transpose() + Q;
        }
        
        void update(const Eigen::Vector3f& measurement) {
            Eigen::Matrix<float, 3, 1> z;
            z << measurement.x(), measurement.y(), measurement.z();
            
            Eigen::Matrix<float, 3, 1> y = z - H * x;
            Eigen::Matrix<float, 3, 3> S = H * P * H.transpose() + R;
            Eigen::Matrix<float, 6, 3> K = P * H.transpose() * S.inverse();
            
            x = x + K * y;
            P = (Eigen::Matrix<float, 6, 6>::Identity() - K * H) * P;
        }
        
        Eigen::Vector3f get_position() {
            return x.head<3>();
        }
        
        Eigen::Vector3f get_velocity() {
            return x.tail<3>();
        }
    };
    
    std::unordered_map<int, ObjectTracker> trackers;
    int next_tracker_id = 0;
    
public:
    LIDARNeuromorphicProcessor() : current_cloud(new pcl::PointCloud<pcl::PointXYZI>) {
        // Initialize neuromorphic components
        NeuronParameters params;
        spatial_neurons = std::make_unique<GridCellNeuron>(params, 1000);
        object_neurons = std::make_unique<PatternNeuron>(params, 500, 10, 128);
        decision_neurons = std::make_unique<BistableNeuron>(params, 10);
        
        // Configure voxel filter
        voxel_filter.setLeafSize(0.1f, 0.1f, 0.1f);  // 10cm voxels
    }
    
    bool initialize(const std::string& pcap_file = "") {
        try {
            if (pcap_file.empty()) {
                // Live Velodyne connection
                velodyne_grabber = std::make_unique<pcl::VLPGrabber>("192.168.1.201");
            } else {
                // Playback from file
                velodyne_grabber = std::make_unique<pcl::VLPGrabber>(pcap_file);
            }
            
            // Register callback
            std::function<void(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr&)> callback = 
                [this](const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& cloud) {
                    process_cloud(cloud);
                };
            
            velodyne_grabber->registerCallback(callback);
            
            return true;
        } catch (const std::exception& e) {
            std::cerr << "LIDAR initialization error: " << e.what() << std::endl;
            return false;
        }
    }
    
    void start() {
        running = true;
        velodyne_grabber->start();
        
        lidar_thread = std::thread([this]() {
            // Set thread priority
            struct sched_param param;
            param.sched_priority = sched_get_priority_max(SCHED_FIFO) - 3;
            pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);
            
            while (running) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        });
    }
    
    void stop() {
        running = false;
        if (velodyne_grabber) {
            velodyne_grabber->stop();
        }
        if (lidar_thread.joinable()) {
            lidar_thread.join();
        }
    }
    
private:
    void process_cloud(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& cloud) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Copy cloud
        *current_cloud = *cloud;
        
        // Downsample
        pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZI>);
        voxel_filter.setInputCloud(current_cloud);
        voxel_filter.filter(*filtered_cloud);
        
        // Segment ground plane
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        
        pcl::SACSegmentation<pcl::PointXYZI> seg;
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(0.1);
        seg.setInputCloud(filtered_cloud);
        seg.segment(*inliers, *coefficients);
        
        // Extract non-ground points
        pcl::PointCloud<pcl::PointXYZI>::Ptr object_cloud(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::ExtractIndices<pcl::PointXYZI> extract;
        extract.setInputCloud(filtered_cloud);
        extract.setIndices(inliers);
        extract.setNegative(true);
        extract.filter(*object_cloud);
        
        // Cluster objects
        std::vector<pcl::PointIndices> cluster_indices;
        euclidean_clustering(object_cloud, cluster_indices);
        
        // Process each cluster through neuromorphic system
        std::vector<DetectedObject> new_objects;
        
        for (const auto& cluster : cluster_indices) {
            if (cluster.indices.size() < 10) continue;  // Skip small clusters
            
            DetectedObject obj = process_cluster_neuromorphic(object_cloud, cluster);
            
            // Update tracker
            update_tracker(obj);
            
            new_objects.push_back(obj);
        }
        
        // Update detected objects
        {
            std::lock_guard<std::mutex> lock(objects_mutex);
            detected_objects = std::move(new_objects);
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        if (duration.count() > 100) {
            std::cerr << "Warning: LIDAR processing took " << duration.count() << "ms" << std::endl;
        }
    }
    
    void euclidean_clustering(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud,
                            std::vector<pcl::PointIndices>& cluster_indices) {
        // KD-Tree for nearest neighbor search
        pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>);
        tree->setInputCloud(cloud);
        
        pcl::EuclideanClusterExtraction<pcl::PointXYZI> ec;
        ec.setClusterTolerance(0.3);  // 30cm
        ec.setMinClusterSize(10);
        ec.setMaxClusterSize(10000);
        ec.setSearchMethod(tree);
        ec.setInputCloud(cloud);
        ec.extract(cluster_indices);
    }
    
    DetectedObject process_cluster_neuromorphic(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud,
                                              const pcl::PointIndices& cluster) {
        DetectedObject obj;
        
        // Compute centroid
        Eigen::Vector4f centroid;
        pcl::compute3DCentroid(*cloud, cluster.indices, centroid);
        obj.position = centroid.head<3>();
        
        // Extract features for neuromorphic processing
        std::vector<float> features = extract_cluster_features(cloud, cluster);
        
        // Convert to neuromorphic input
        std::vector<double> neuron_input(features.size());
        std::transform(features.begin(), features.end(), neuron_input.begin(),
                      [](float f) { return static_cast<double>(f); });
        
        // Process through pattern neurons
        object_neurons->process_input_pattern(neuron_input.data(), features.size());
        
        // Run neuron dynamics
        std::vector<double> voltages(500, -65.0);
        std::vector<double> adaptations(500, 0.0);
        std::vector<bool> spikes(500);
        
        object_neurons->update_state(voltages.data(), adaptations.data(),
                                   neuron_input.data(), 500, 0.1);
        object_neurons->check_threshold(voltages.data(), spikes.data(), 500);
        
        // Count spikes per pattern type
        std::vector<int> pattern_votes(10, 0);
        for (int i = 0; i < 500; ++i) {
            if (spikes[i]) {
                pattern_votes[i % 10]++;
            }
        }
        
        // Find winning pattern
        auto max_it = std::max_element(pattern_votes.begin(), pattern_votes.end());
        obj.object_class = std::distance(pattern_votes.begin(), max_it);
        obj.confidence = static_cast<float>(*max_it) / 50.0f;
        
        // Store neuromorphic signature
        obj.neuromorphic_signature.resize(10);
        for (int i = 0; i < 10; ++i) {
            obj.neuromorphic_signature[i] = static_cast<float>(pattern_votes[i]) / 50.0f;
        }
        
        return obj;
    }
    
    std::vector<float> extract_cluster_features(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud,
                                               const pcl::PointIndices& cluster) {
        std::vector<float> features;
        
        // Basic geometric features
        Eigen::Vector4f min_pt, max_pt;
        pcl::getMinMax3D(*cloud, cluster, min_pt, max_pt);
        
        features.push_back(max_pt.x() - min_pt.x());  // Width
        features.push_back(max_pt.y() - min_pt.y());  // Depth
        features.push_back(max_pt.z() - min_pt.z());  // Height
        
        // Point density
        float volume = features[0] * features[1] * features[2];
        features.push_back(cluster.indices.size() / volume);
        
        // Intensity statistics
        float mean_intensity = 0;
        float max_intensity = 0;
        for (int idx : cluster.indices) {
            mean_intensity += cloud->points[idx].intensity;
            max_intensity = std::max(max_intensity, cloud->points[idx].intensity);
        }
        mean_intensity /= cluster.indices.size();
        
        features.push_back(mean_intensity);
        features.push_back(max_intensity);
        
        // Compute normals and curvature
        pcl::NormalEstimation<pcl::PointXYZI, pcl::Normal> ne;
        pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
        pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>());
        
        ne.setInputCloud(cloud);
        ne.setIndices(boost::make_shared<pcl::PointIndices>(cluster));
        ne.setSearchMethod(tree);
        ne.setKSearch(10);
        ne.compute(*normals);
        
        // Average curvature
        float avg_curvature = 0;
        for (const auto& normal : normals->points) {
            avg_curvature += normal.curvature;
        }
        avg_curvature /= normals->points.size();
        features.push_back(avg_curvature);
        
        // Pad to 128 features for neural network
        while (features.size() < 128) {
            features.push_back(0.0f);
        }
        
        return features;
    }
    
    void update_tracker(DetectedObject& obj) {
        // Find nearest existing tracker
        int best_tracker = -1;
        float min_distance = 5.0f;  // 5 meter threshold
        
        for (auto& [id, tracker] : trackers) {
            tracker.predict();
            float dist = (tracker.get_position() - obj.position).norm();
            if (dist < min_distance) {
                min_distance = dist;
                best_tracker = id;
            }
        }
        
        if (best_tracker >= 0) {
            // Update existing tracker
            trackers[best_tracker].update(obj.position);
            obj.velocity = trackers[best_tracker].get_velocity();
        } else {
            // Create new tracker
            trackers[next_tracker_id] = ObjectTracker();
            trackers[next_tracker_id].update(obj.position);
            obj.velocity = Eigen::Vector3f::Zero();
            next_tracker_id++;
        }
    }
    
public:
    std::vector<DetectedObject> get_detected_objects() {
        std::lock_guard<std::mutex> lock(objects_mutex);
        return detected_objects;
    }
    
    // Spatial query using neuromorphic grid cells
    std::vector<int> query_spatial_region(float x, float y, float radius) {
        std::vector<int> nearby_objects;
        
        // Activate grid cells for the query region
        std::vector<double> grid_input(1000);
        spatial_neurons->compute_grid_input(x, y, grid_input.data(), 1000);
        
        // Run grid cell dynamics
        std::vector<double> voltages(1000, -65.0);
        std::vector<double> adaptations(1000, 0.0);
        std::vector<bool> spikes(1000);
        
        spatial_neurons->update_state(voltages.data(), adaptations.data(),
                                    grid_input.data(), 1000, 0.1);
        spatial_neurons->check_threshold(voltages.data(), spikes.data(), 1000);
        
        // Check which objects activate similar grid cells
        std::lock_guard<std::mutex> lock(objects_mutex);
        for (size_t i = 0; i < detected_objects.size(); ++i) {
            const auto& obj = detected_objects[i];
            
            // Check if object is within radius
            float dx = obj.position.x() - x;
            float dy = obj.position.y() - y;
            if (dx*dx + dy*dy <= radius*radius) {
                nearby_objects.push_back(i);
            }
        }
        
        return nearby_objects;
    }
};

/**
 * Audio processor with neuromorphic speech/sound recognition
 */
class AudioNeuromorphicProcessor {
private:
    // Audio components
    PaStream* pa_stream = nullptr;
    SNDFILE* audio_file = nullptr;
    SF_INFO audio_info;
    
    // Audio parameters
    static constexpr int SAMPLE_RATE = 48000;
    static constexpr int FRAMES_PER_BUFFER = 512;
    static constexpr int NUM_CHANNELS = 2;
    
    // Neuromorphic components
    std::unique_ptr<ResonatorNeuron> frequency_analyzer;
    std::unique_ptr<BurstNeuron> rhythm_detector;
    std::unique_ptr<PatternNeuron> speech_recognizer;
    
    // FFT for spectral analysis
    fftwf_plan fft_plan;
    std::vector<float> fft_input;
    std::vector<std::complex<float>> fft_output;
    
    // Mel-frequency cepstral coefficients
    class MFCCProcessor {
    private:
        int num_filters = 40;
        int num_coeffs = 13;
        std::vector<std::vector<float>> mel_filterbank;
        std::vector<float> mel_energies;
        fftwf_plan dct_plan;
        
    public:
        MFCCProcessor(int sample_rate, int fft_size) {
            // Create mel filterbank
            create_mel_filterbank(sample_rate, fft_size);
            mel_energies.resize(num_filters);
            
            // DCT for cepstral coefficients
            dct_plan = fftwf_plan_r2r_1d(num_filters, mel_energies.data(), 
                                        mel_energies.data(), FFTW_REDFT10, FFTW_MEASURE);
        }
        
        ~MFCCProcessor() {
            fftwf_destroy_plan(dct_plan);
        }
        
        void create_mel_filterbank(int sample_rate, int fft_size) {
            mel_filterbank.resize(num_filters);
            
            float mel_low = 2595 * log10(1 + 300.0f / 700);
            float mel_high = 2595 * log10(1 + (sample_rate / 2.0f) / 700);
            
            std::vector<float> mel_points(num_filters + 2);
            for (int i = 0; i < num_filters + 2; ++i) {
                float mel = mel_low + i * (mel_high - mel_low) / (num_filters + 1);
                float freq = 700 * (pow(10, mel / 2595) - 1);
                mel_points[i] = (fft_size + 1) * freq / sample_rate;
            }
            
            for (int i = 0; i < num_filters; ++i) {
                mel_filterbank[i].resize(fft_size / 2 + 1, 0.0f);
                
                for (int j = 0; j < fft_size / 2 + 1; ++j) {
                    if (j >= mel_points[i] && j <= mel_points[i + 1]) {
                        mel_filterbank[i][j] = (j - mel_points[i]) / 
                                              (mel_points[i + 1] - mel_points[i]);
                    } else if (j >= mel_points[i + 1] && j <= mel_points[i + 2]) {
                        mel_filterbank[i][j] = (mel_points[i + 2] - j) / 
                                              (mel_points[i + 2] - mel_points[i + 1]);
                    }
                }
            }
        }
        
        std::vector<float> compute_mfcc(const std::vector<float>& spectrum) {
            // Apply mel filterbank
            for (int i = 0; i < num_filters; ++i) {
                mel_energies[i] = 0;
                for (size_t j = 0; j < spectrum.size(); ++j) {
                    mel_energies[i] += spectrum[j] * mel_filterbank[i][j];
                }
                mel_energies[i] = log(mel_energies[i] + 1e-10f);
            }
            
            // Apply DCT
            fftwf_execute(dct_plan);
            
            // Return first num_coeffs
            std::vector<float> mfcc(num_coeffs);
            std::copy(mel_energies.begin(), mel_energies.begin() + num_coeffs, mfcc.begin());
            
            return mfcc;
        }
    };
    
    std::unique_ptr<MFCCProcessor> mfcc_processor;
    
    // Audio buffer
    LockFreeRingBuffer<float, 48000 * 10> audio_buffer;  // 10 seconds
    
    // Voice activity detection
    class VADProcessor {
    private:
        float energy_threshold = 0.01f;
        float zcr_threshold = 0.1f;
        int hangover_time = 10;
        int hangover_counter = 0;
        bool is_speech = false;
        
    public:
        bool process_frame(const float* frame, int frame_size) {
            // Compute energy
            float energy = 0;
            for (int i = 0; i < frame_size; ++i) {
                energy += frame[i] * frame[i];
            }
            energy /= frame_size;
            
            // Compute zero crossing rate
            int zero_crossings = 0;
            for (int i = 1; i < frame_size; ++i) {
                if ((frame[i-1] >= 0 && frame[i] < 0) || 
                    (frame[i-1] < 0 && frame[i] >= 0)) {
                    zero_crossings++;
                }
            }
            float zcr = static_cast<float>(zero_crossings) / frame_size;
            
            // Speech detection logic
            if (energy > energy_threshold && zcr < zcr_threshold) {
                is_speech = true;
                hangover_counter = hangover_time;
            } else if (hangover_counter > 0) {
                hangover_counter--;
            } else {
                is_speech = false;
            }
            
            return is_speech;
        }
        
        void set_thresholds(float energy_thresh, float zcr_thresh) {
            energy_threshold = energy_thresh;
            zcr_threshold = zcr_thresh;
        }
    };
    
    VADProcessor vad;
    
    // Recognition results
    struct AudioEvent {
        float timestamp;
        std::string event_type;  // "speech", "music", "noise", "alarm", etc.
        float confidence;
        std::vector<float> neuromorphic_features;
    };
    
    std::vector<AudioEvent> detected_events;
    std::mutex events_mutex;
    
public:
    AudioNeuromorphicProcessor() {
        // Initialize neuromorphic components
        NeuronParameters params;
        frequency_analyzer = std::make_unique<ResonatorNeuron>(params, 256);
        rhythm_detector = std::make_unique<BurstNeuron>(params, 64);
        speech_recognizer = std::make_unique<PatternNeuron>(params, 200, 20, 13);
        
        // Initialize FFT
        fft_input.resize(FRAMES_PER_BUFFER);
        fft_output.resize(FRAMES_PER_BUFFER / 2 + 1);
        fft_plan = fftwf_plan_dft_r2c_1d(FRAMES_PER_BUFFER,
                                        fft_input.data(),
                                        reinterpret_cast<fftwf_complex*>(fft_output.data()),
                                        FFTW_MEASURE);
        
        // Initialize MFCC processor
        mfcc_processor = std::make_unique<MFCCProcessor>(SAMPLE_RATE, FRAMES_PER_BUFFER);
        
        // Initialize PortAudio
        Pa_Initialize();
    }
    
    ~AudioNeuromorphicProcessor() {
        stop();
        Pa_Terminate();
        fftwf_destroy_plan(fft_plan);
        if (audio_file) {
            sf_close(audio_file);
        }
    }
    
    bool initialize_input(const std::string& device_name = "") {
        PaStreamParameters inputParameters;
        inputParameters.device = Pa_GetDefaultInputDevice();
        
        if (!device_name.empty()) {
            // Find device by name
            int numDevices = Pa_GetDeviceCount();
            for (int i = 0; i < numDevices; ++i) {
                const PaDeviceInfo* deviceInfo = Pa_GetDeviceInfo(i);
                if (deviceInfo && std::string(deviceInfo->name).find(device_name) != std::string::npos) {
                    inputParameters.device = i;
                    break;
                }
            }
        }
        
        if (inputParameters.device == paNoDevice) {
            std::cerr << "No audio input device found" << std::endl;
            return false;
        }
        
        inputParameters.channelCount = NUM_CHANNELS;
        inputParameters.sampleFormat = paFloat32;
        inputParameters.suggestedLatency = Pa_GetDeviceInfo(inputParameters.device)->defaultLowInputLatency;
        inputParameters.hostApiSpecificStreamInfo = nullptr;
        
        PaError err = Pa_OpenStream(&pa_stream,
                                  &inputParameters,
                                  nullptr,  // No output
                                  SAMPLE_RATE,
                                  FRAMES_PER_BUFFER,
                                  paClipOff,
                                  audio_callback,
                                  this);
        
        if (err != paNoError) {
            std::cerr << "PortAudio error: " << Pa_GetErrorText(err) << std::endl;
            return false;
        }
        
        return true;
    }
    
    bool initialize_file(const std::string& filename) {
        audio_info.format = 0;
        audio_file = sf_open(filename.c_str(), SFM_READ, &audio_info);
        
        if (!audio_file) {
            std::cerr << "Failed to open audio file: " << filename << std::endl;
            return false;
        }
        
        std::cout << "Audio file: " << filename << std::endl;
        std::cout << "Sample rate: " << audio_info.samplerate << std::endl;
        std::cout << "Channels: " << audio_info.channels << std::endl;
        std::cout << "Frames: " << audio_info.frames << std::endl;
        
        return true;
    }
    
    static int audio_callback(const void* inputBuffer, void* outputBuffer,
                            unsigned long framesPerBuffer,
                            const PaStreamCallbackTimeInfo* timeInfo,
                            PaStreamCallbackFlags statusFlags,
                            void* userData) {
        auto* processor = static_cast<AudioNeuromorphicProcessor*>(userData);
        const float* input = static_cast<const float*>(inputBuffer);
        
        // Write to ring buffer
        processor->audio_buffer.write(input, framesPerBuffer * NUM_CHANNELS);
        
        return paContinue;
    }
    
    void start() {
        if (pa_stream) {
            Pa_StartStream(pa_stream);
        }
        
        // Start processing thread
        std::thread processing_thread([this]() {
            process_audio_stream();
        });
        processing_thread.detach();
    }
    
    void stop() {
        if (pa_stream) {
            Pa_CloseStream(pa_stream);
            pa_stream = nullptr;
        }
    }
    
private:
    void process_audio_stream() {
        float frame_buffer[FRAMES_PER_BUFFER * NUM_CHANNELS];
        
        while (true) {
            if (audio_buffer.read(frame_buffer, FRAMES_PER_BUFFER * NUM_CHANNELS)) {
                // Convert stereo to mono
                for (int i = 0; i < FRAMES_PER_BUFFER; ++i) {
                    fft_input[i] = (frame_buffer[i * 2] + frame_buffer[i * 2 + 1]) / 2.0f;
                }
                
                // Apply window function
                apply_hamming_window(fft_input.data(), FRAMES_PER_BUFFER);
                
                // Compute FFT
                fftwf_execute(fft_plan);
                
                // Compute magnitude spectrum
                std::vector<float> spectrum(FRAMES_PER_BUFFER / 2 + 1);
                for (int i = 0; i <= FRAMES_PER_BUFFER / 2; ++i) {
                    spectrum[i] = std::abs(fft_output[i]);
                }
                
                // Voice activity detection
                bool is_speech = vad.process_frame(fft_input.data(), FRAMES_PER_BUFFER);
                
                if (is_speech) {
                    // Compute MFCC features
                    auto mfcc = mfcc_processor->compute_mfcc(spectrum);
                    
                    // Process through neuromorphic network
                    process_audio_neuromorphic(spectrum, mfcc);
                }
                
                // Rhythm detection
                detect_rhythm_patterns(fft_input.data(), FRAMES_PER_BUFFER);
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }
    
    void apply_hamming_window(float* data, int size) {
        for (int i = 0; i < size; ++i) {
            float window = 0.54f - 0.46f * cos(2.0f * M_PI * i / (size - 1));
            data[i] *= window;
        }
    }
    
    void process_audio_neuromorphic(const std::vector<float>& spectrum,
                                  const std::vector<float>& mfcc) {
        // Frequency analysis with resonator neurons
        std::vector<double> freq_input(256);
        
        // Downsample spectrum to neuromorphic resolution
        int bins_per_neuron = spectrum.size() / 256;
        for (int i = 0; i < 256; ++i) {
            double sum = 0;
            for (int j = 0; j < bins_per_neuron; ++j) {
                sum += spectrum[i * bins_per_neuron + j];
            }
            freq_input[i] = sum / bins_per_neuron;
        }
        
        // Process through frequency analyzer
        std::vector<double> freq_voltages(256, -65.0);
        std::vector<double> freq_adapt(256, 0.0);
        std::vector<bool> freq_spikes(256);
        
        frequency_analyzer->update_state(freq_voltages.data(), freq_adapt.data(),
                                       freq_input.data(), 256, 0.1);
        frequency_analyzer->check_threshold(freq_voltages.data(), freq_spikes.data(), 256);
        
        // Get dominant frequencies
        auto dominant_freqs = frequency_analyzer->get_dominant_frequencies(
            freq_voltages.data(), 256, SAMPLE_RATE);
        
        // Speech recognition with pattern neurons
        std::vector<double> mfcc_input(mfcc.size());
        std::transform(mfcc.begin(), mfcc.end(), mfcc_input.begin(),
                      [](float f) { return static_cast<double>(f); });
        
        speech_recognizer->process_input_pattern(mfcc_input.data(), mfcc.size());
        
        // Run pattern recognition
        std::vector<double> pattern_voltages(200, -65.0);
        std::vector<double> pattern_adapt(200, 0.0);
        std::vector<bool> pattern_spikes(200);
        
        speech_recognizer->update_state(pattern_voltages.data(), pattern_adapt.data(),
                                      mfcc_input.data(), 200, 0.1);
        speech_recognizer->check_threshold(pattern_voltages.data(), 
                                         pattern_spikes.data(), 200);
        
        // Classify audio event
        AudioEvent event;
        event.timestamp = Pa_GetStreamTime(pa_stream);
        event.confidence = 0.0f;
        
        // Count pattern activations
        std::vector<int> pattern_counts(20, 0);
        for (int i = 0; i < 200; ++i) {
            if (pattern_spikes[i]) {
                pattern_counts[i % 20]++;
            }
        }
        
        // Determine event type based on pattern
        int max_pattern = std::distance(pattern_counts.begin(),
                                      std::max_element(pattern_counts.begin(), 
                                                     pattern_counts.end()));
        
        // Map patterns to event types
        switch (max_pattern) {
            case 0: case 1: case 2: 
                event.event_type = "speech"; 
                break;
            case 3: case 4: case 5: 
                event.event_type = "music"; 
                break;
            case 6: case 7: 
                event.event_type = "alarm"; 
                break;
            case 8: case 9: 
                event.event_type = "vehicle"; 
                break;
            case 10: case 11: 
                event.event_type = "gunshot"; 
                break;
            case 12: case 13: 
                event.event_type = "explosion"; 
                break;
            default: 
                event.event_type = "unknown"; 
                break;
        }
        
        event.confidence = static_cast<float>(pattern_counts[max_pattern]) / 10.0f;
        
        // Store neuromorphic features
        event.neuromorphic_features.resize(20);
        for (int i = 0; i < 20; ++i) {
            event.neuromorphic_features[i] = static_cast<float>(pattern_counts[i]) / 10.0f;
        }
        
        // Add to detected events if confidence is high enough
        if (event.confidence > 0.5f) {
            std::lock_guard<std::mutex> lock(events_mutex);
            detected_events.push_back(event);
            
            // Keep only recent events (last 100)
            if (detected_events.size() > 100) {
                detected_events.erase(detected_events.begin());
            }
        }
    }
    
    void detect_rhythm_patterns(const float* audio_frame, int frame_size) {
        // Energy envelope for rhythm detection
        float frame_energy = 0;
        for (int i = 0; i < frame_size; ++i) {
            frame_energy += audio_frame[i] * audio_frame[i];
        }
        frame_energy = sqrt(frame_energy / frame_size);
        
        // Feed energy to burst neurons for rhythm detection
        std::vector<double> rhythm_input(64, frame_energy * 10.0);
        std::vector<double> rhythm_voltages(64, -65.0);
        std::vector<double> rhythm_adapt(64, 0.0);
        std::vector<bool> rhythm_spikes(64);
        
        rhythm_detector->update_state(rhythm_voltages.data(), rhythm_adapt.data(),
                                    rhythm_input.data(), 64, 0.1);
        rhythm_detector->check_threshold(rhythm_voltages.data(), 
                                       rhythm_spikes.data(), 64);
        
        // Analyze burst patterns for tempo detection
        auto burst_counts = rhythm_detector->get_burst_counts(64);
        
        // Simple tempo estimation from burst patterns
        int total_bursts = 0;
        for (int count : burst_counts) {
            total_bursts += count;
        }
        
        if (total_bursts > 10) {
            float estimated_bpm = (total_bursts * 60.0f) / (frame_size / static_cast<float>(SAMPLE_RATE));
            
            // Log tempo detection
            if (estimated_bpm > 60 && estimated_bpm < 200) {
                // Detected musical tempo
                AudioEvent event;
                event.timestamp = Pa_GetStreamTime(pa_stream);
                event.event_type = "tempo_" + std::to_string(static_cast<int>(estimated_bpm));
                event.confidence = 0.7f;
                
                std::lock_guard<std::mutex> lock(events_mutex);
                detected_events.push_back(event);
            }
        }
    }
    
public:
    std::vector<AudioEvent> get_recent_events() {
        std::lock_guard<std::mutex> lock(events_mutex);
        return detected_events;
    }
    
    // Real-time audio feature extraction for AR visualization
    struct AudioFeatures {
        std::vector<float> spectrum;
        std::vector<float> mfcc;
        float energy;
        float zero_crossing_rate;
        bool is_speech;
        float tempo_bpm;
    };
    
    AudioFeatures get_current_features() {
        AudioFeatures features;
        
        // Get latest frame
        float frame[FRAMES_PER_BUFFER];
        if (audio_buffer.read(frame, FRAMES_PER_BUFFER)) {
            // Compute spectrum
            std::copy(frame, frame + FRAMES_PER_BUFFER, fft_input.begin());
            apply_hamming_window(fft_input.data(), FRAMES_PER_BUFFER);
            fftwf_execute(fft_plan);
            
            features.spectrum.resize(FRAMES_PER_BUFFER / 2 + 1);
            for (int i = 0; i <= FRAMES_PER_BUFFER / 2; ++i) {
                features.spectrum[i] = 20 * log10(std::abs(fft_output[i]) + 1e-6f);
            }
            
            // Compute MFCC
            features.mfcc = mfcc_processor->compute_mfcc(features.spectrum);
            
            // Energy and ZCR
            features.energy = 0;
            int zcr = 0;
            for (int i = 0; i < FRAMES_PER_BUFFER; ++i) {
                features.energy += frame[i] * frame[i];
                if (i > 0 && ((frame[i-1] >= 0 && frame[i] < 0) || 
                             (frame[i-1] < 0 && frame[i] >= 0))) {
                    zcr++;
                }
            }
            features.energy = sqrt(features.energy / FRAMES_PER_BUFFER);
            features.zero_crossing_rate = static_cast<float>(zcr) / FRAMES_PER_BUFFER;
            
            features.is_speech = vad.process_frame(frame, FRAMES_PER_BUFFER);
        }
        
        return features;
    }
};

} // namespace neuromorphic
} // namespace ares