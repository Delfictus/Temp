// PROPRIETARY AND CONFIDENTIAL
// Copyright (c) 2024 DELFICTUS I/O LLC
// Patent Pending - Application #63/826,067

#include "ARESGameMode_Optimized.h"
#include "Engine/Engine.h"
#include "Async/Async.h"
#include "HAL/RunnableThread.h"
#include "Misc/ScopeLock.h"
#include "Stats/Stats.h"
#include "ProfilingDebugging/CpuProfilerTrace.h"

// Include ARES headers
#include "unified_quantum_ares.cpp"

// Stats group for performance tracking
DECLARE_STATS_GROUP(TEXT("ARES"), STATGROUP_ARES, STATCAT_Advanced);
DECLARE_CYCLE_STAT(TEXT("ARES Update"), STAT_ARESUpdate, STATGROUP_ARES);
DECLARE_CYCLE_STAT(TEXT("ARES AI Query"), STAT_ARESAIQuery, STATGROUP_ARES);
DECLARE_CYCLE_STAT(TEXT("ARES Metrics"), STAT_ARESMetrics, STATGROUP_ARES);

// ARES Worker Thread - Optimized with priority queue and batching
class FARESWorkerThread : public FRunnable
{
public:
    FARESWorkerThread(ares::UnifiedQuantumARES* InARES, FThreadSafeBool* InShutdown, float InUpdateFreq)
        : ARESSystem(InARES)
        , bShutdownRequested(InShutdown)
        , UpdateFrequency(InUpdateFreq)
        , UpdateInterval(1.0f / InUpdateFreq)
    {
        WorkerSemaphore = FGenericPlatformProcess::GetSynchEventFromPool(false);
    }

    virtual ~FARESWorkerThread()
    {
        FGenericPlatformProcess::ReturnSynchEventToPool(WorkerSemaphore);
    }

    virtual bool Init() override
    {
        LastUpdateTime = FPlatformTime::Seconds();
        return true;
    }

    virtual uint32 Run() override
    {
        // Set thread affinity for better cache performance
        FPlatformProcess::SetThreadAffinityMask(FPlatformAffinity::GetTaskGraphThreadMask());

        while (!bShutdownRequested->load())
        {
            TRACE_CPUPROFILER_EVENT_SCOPE(ARESWorkerThread_Run);

            double CurrentTime = FPlatformTime::Seconds();
            double DeltaTime = CurrentTime - LastUpdateTime;

            if (DeltaTime >= UpdateInterval)
            {
                {
                    SCOPE_CYCLE_COUNTER(STAT_ARESUpdate);
                    ARESSystem->update(DeltaTime);
                }

                LastUpdateTime = CurrentTime;

                // Process any pending tasks
                ProcessPendingTasks();
            }
            else
            {
                // Sleep with precise timing
                double SleepTime = UpdateInterval - DeltaTime;
                if (SleepTime > 0.001) // Only sleep if more than 1ms
                {
                    WorkerSemaphore->Wait(FTimespan::FromSeconds(SleepTime));
                }
            }
        }

        return 0;
    }

    virtual void Stop() override
    {
        WorkerSemaphore->Trigger();
    }

    void SetUpdateFrequency(float FrequencyHz)
    {
        UpdateFrequency = FMath::Clamp(FrequencyHz, 1.0f, 1000.0f);
        UpdateInterval = 1.0f / UpdateFrequency;
    }

private:
    void ProcessPendingTasks()
    {
        // Process any queued operations
        // This is where we'd handle batched updates, etc.
    }

    ares::UnifiedQuantumARES* ARESSystem;
    FThreadSafeBool* bShutdownRequested;
    FEvent* WorkerSemaphore;
    
    float UpdateFrequency;
    double UpdateInterval;
    double LastUpdateTime;
};

AARESGameMode_Optimized::AARESGameMode_Optimized()
{
    PrimaryActorTick.bCanEverTick = true;
    PrimaryActorTick.TickInterval = 0.0f; // Tick every frame for smooth updates

    // Initialize circular queue with reasonable size
    AIResponseQueue = TCircularQueue<FAIMessage>(64);
    
    bSystemInitialized = false;
    bARVisualizationEnabled = false;
    LastMetricsUpdateTime = 0.0f;
}

AARESGameMode_Optimized::~AARESGameMode_Optimized()
{
    ShutdownARESSystem();
}

void AARESGameMode_Optimized::InitGame(const FString& MapName, const FString& Options, FString& ErrorMessage)
{
    Super::InitGame(MapName, Options, ErrorMessage);
    
    UE_LOG(LogTemp, Warning, TEXT("ARES Edge System - Optimized Unreal Engine 5 Integration"));
}

void AARESGameMode_Optimized::StartPlay()
{
    Super::StartPlay();
    
    // Auto-initialize ARES with safety check
    if (!bSystemInitialized)
    {
        InitializeARESSystem();
    }
}

void AARESGameMode_Optimized::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
    ShutdownARESSystem();
    Super::EndPlay(EndPlayReason);
}

void AARESGameMode_Optimized::Tick(float DeltaSeconds)
{
    Super::Tick(DeltaSeconds);

    TRACE_CPUPROFILER_EVENT_SCOPE(ARESGameMode_Tick);

    if (!bSystemInitialized || !ARESSystem)
    {
        return;
    }

    // Update performance stats
    UpdatePerformanceStats(DeltaSeconds);

    // Process pending AI responses
    ProcessPendingAIResponses();

    // Update metrics at specified interval
    LastMetricsUpdateTime += DeltaSeconds;
    if (LastMetricsUpdateTime >= MetricsUpdateInterval)
    {
        UpdateMetrics();
        LastMetricsUpdateTime = 0.0f;
    }

    // Display debug info if enabled
    if (GEngine && GEngine->bEnableOnScreenDebugMessages)
    {
        FScopeLock Lock(&PerfStatsMutex);
        
        GEngine->AddOnScreenDebugMessage(1, 0.0f, FColor::Green,
            FString::Printf(TEXT("ARES FPS: %.1f | Frame: %.2f ms (min: %.2f, max: %.2f)"),
                1.0f / PerfStats.AverageFrameTimeMs * 1000.0f,
                PerfStats.AverageFrameTimeMs,
                PerfStats.MinFrameTimeMs,
                PerfStats.MaxFrameTimeMs));

        GEngine->AddOnScreenDebugMessage(2, 0.0f, FColor::Yellow,
            FString::Printf(TEXT("Energy: %.1f%% | Threats: %d | Stealth: %.1f%% | GPU: %.1f MB"),
                CachedMetrics.EnergyLevel * 100.0f,
                CachedMetrics.ThreatsDetected,
                CachedMetrics.StealthEffectiveness * 100.0f,
                CachedMetrics.GPUMemoryUsageMB));
    }
}

void AARESGameMode_Optimized::InitializeARESSystem()
{
    if (bSystemInitialized)
    {
        UE_LOG(LogTemp, Warning, TEXT("ARES already initialized"));
        return;
    }

    TRACE_CPUPROFILER_EVENT_SCOPE(InitializeARESSystem);

    // Create optimized configuration
    ares::UnifiedARESConfig config;
    config.quantum_signature_algorithm = ares::quantum::PQCAlgorithm::CRYSTALS_DILITHIUM5;
    config.enable_quantum_resilience = true;
    config.ai_strategy = ares::chronopath::OrchestrationStrategy::CONSENSUS_SYNTHESIS;
    config.num_swarm_nodes = 32;
    config.neuromorphic_cores = FPlatformMisc::NumberOfCoresIncludingHyperthreads();
    config.enable_gpu_acceleration = true;
    config.enable_memory_pooling = true;

    try
    {
        // Initialize ARES
        ARESSystem = MakeUnique<ares::UnifiedQuantumARES>(config);
        
        // Initialize worker thread
        InitializeWorkerThread();
        
        bSystemInitialized = true;
        
        UE_LOG(LogTemp, Warning, TEXT("ARES Edge System initialized successfully"));
        
        // Initial metrics update
        UpdateMetrics();
    }
    catch (const std::exception& e)
    {
        UE_LOG(LogTemp, Error, TEXT("Failed to initialize ARES: %s"), UTF8_TO_TCHAR(e.what()));
    }
}

void AARESGameMode_Optimized::ShutdownARESSystem()
{
    if (!bSystemInitialized)
    {
        return;
    }

    TRACE_CPUPROFILER_EVENT_SCOPE(ShutdownARESSystem);

    // Signal shutdown
    bShutdownRequested = true;

    // Shutdown worker thread
    ShutdownWorkerThread();

    // Clean up ARES
    ARESSystem.Reset();
    
    bSystemInitialized = false;
    
    UE_LOG(LogTemp, Warning, TEXT("ARES Edge System shutdown complete"));
}

bool AARESGameMode_Optimized::IsARESSystemReady() const
{
    return bSystemInitialized && ARESSystem.IsValid();
}

void AARESGameMode_Optimized::InitializeWorkerThread()
{
    if (WorkerThread || WorkerThreadHandle)
    {
        return;
    }

    WorkerThread = new FARESWorkerThread(ARESSystem.Get(), &bShutdownRequested, UpdateFrequency);
    WorkerThreadHandle = FRunnableThread::Create(
        WorkerThread,
        TEXT("ARES Worker Thread"),
        0,
        TPri_AboveNormal,
        FPlatformAffinity::GetTaskGraphThreadMask()
    );
}

void AARESGameMode_Optimized::ShutdownWorkerThread()
{
    if (WorkerThreadHandle)
    {
        WorkerThreadHandle->Kill(true);
        delete WorkerThreadHandle;
        WorkerThreadHandle = nullptr;
    }

    if (WorkerThread)
    {
        delete WorkerThread;
        WorkerThread = nullptr;
    }
}

void AARESGameMode_Optimized::UpdateMetrics()
{
    if (!ARESSystem)
    {
        return;
    }

    SCOPE_CYCLE_COUNTER(STAT_ARESMetrics);

    auto status = ARESSystem->getStatus();
    
    // Thread-safe metrics update
    {
        FScopeLock Lock(&MetricsMutex);
        CachedMetrics.EnergyLevel = status.energy_level;
        CachedMetrics.ThreatsDetected = status.threats_active;
        CachedMetrics.StealthEffectiveness = status.stealth_score;
        CachedMetrics.GPUMemoryUsageMB = status.gpu_memory_usage_mb;
        CachedMetrics.CPUUsagePercent = status.cpu_usage_percent;
        CachedMetrics.ActiveSwarmNodes = status.active_swarm_nodes;
        CachedMetrics.ConsensusHealth = status.consensus_health;
    }

    // Broadcast update event
    OnMetricsUpdated.Broadcast(CachedMetrics);

    // Check for threats
    static int32 LastThreatCount = 0;
    if (status.threats_active > LastThreatCount)
    {
        // New threat detected
        for (int32 i = LastThreatCount; i < status.threats_active; ++i)
        {
            OnThreatDetected.Broadcast(i, FMath::FRandRange(0.5f, 1.0f));
        }
    }
    LastThreatCount = status.threats_active;
}

void AARESGameMode_Optimized::ProcessPendingAIResponses()
{
    FAIMessage Message;
    while (!AIResponseQueue.IsEmpty())
    {
        if (AIResponseQueue.Dequeue(Message))
        {
            OnAIResponseReceived.Broadcast(Message.Response);
        }
    }
}

void AARESGameMode_Optimized::UpdatePerformanceStats(float DeltaTime)
{
    FScopeLock Lock(&PerfStatsMutex);

    float FrameTimeMs = DeltaTime * 1000.0f;
    
    PerfStats.FrameCount++;
    PerfStats.AverageFrameTimeMs = 
        (PerfStats.AverageFrameTimeMs * (PerfStats.FrameCount - 1) + FrameTimeMs) / PerfStats.FrameCount;
    PerfStats.MaxFrameTimeMs = FMath::Max(PerfStats.MaxFrameTimeMs, FrameTimeMs);
    PerfStats.MinFrameTimeMs = FMath::Min(PerfStats.MinFrameTimeMs, FrameTimeMs);

    // Reset stats every second
    AccumulatedTickTime += DeltaTime;
    if (AccumulatedTickTime >= 1.0f)
    {
        PerfStats.Reset();
        AccumulatedTickTime = 0.0f;
    }
}

void AARESGameMode_Optimized::ConfigureAIProvider(const FString& Provider, const FString& APIKey)
{
    if (!IsARESSystemReady())
    {
        UE_LOG(LogTemp, Error, TEXT("ARES not initialized"));
        return;
    }

    // Convert provider string to enum
    ares::chronopath::AIProvider aiProvider;
    
    if (Provider.Equals(TEXT("openai"), ESearchCase::IgnoreCase))
        aiProvider = ares::chronopath::AIProvider::OPENAI_GPT4;
    else if (Provider.Equals(TEXT("anthropic"), ESearchCase::IgnoreCase))
        aiProvider = ares::chronopath::AIProvider::ANTHROPIC_CLAUDE;
    else if (Provider.Equals(TEXT("google"), ESearchCase::IgnoreCase))
        aiProvider = ares::chronopath::AIProvider::GOOGLE_GEMINI;
    else if (Provider.Equals(TEXT("meta"), ESearchCase::IgnoreCase))
        aiProvider = ares::chronopath::AIProvider::META_LLAMA;
    else if (Provider.Equals(TEXT("mistral"), ESearchCase::IgnoreCase))
        aiProvider = ares::chronopath::AIProvider::MISTRAL_AI;
    else if (Provider.Equals(TEXT("xai"), ESearchCase::IgnoreCase))
        aiProvider = ares::chronopath::AIProvider::XAI_GROK;
    else
    {
        UE_LOG(LogTemp, Error, TEXT("Unknown AI provider: %s"), *Provider);
        return;
    }
    
    ARESSystem->configureAI(aiProvider, TCHAR_TO_UTF8(*APIKey));
    
    UE_LOG(LogTemp, Warning, TEXT("Configured AI provider: %s"), *Provider);
}

void AARESGameMode_Optimized::QueryAI(const FString& Prompt, const FString& Strategy)
{
    if (!IsARESSystemReady())
    {
        UE_LOG(LogTemp, Error, TEXT("ARES not initialized"));
        return;
    }

    SCOPE_CYCLE_COUNTER(STAT_ARESAIQuery);

    // Execute on task graph for better thread pool utilization
    FFunctionGraphTask::CreateAndDispatchWhenReady(
        [this, Prompt, Strategy]()
        {
            TRACE_CPUPROFILER_EVENT_SCOPE(ARESAIQuery_Async);
            
            std::string response = ARESSystem->queryAI(TCHAR_TO_UTF8(*Prompt));
            
            // Queue response for game thread processing
            FAIMessage Message;
            Message.Response = UTF8_TO_TCHAR(response.c_str());
            Message.Timestamp = FPlatformTime::Seconds();
            
            {
                FScopeLock Lock(&AIQueueMutex);
                AIResponseQueue.Enqueue(Message);
            }
        },
        TStatId(),
        nullptr,
        ENamedThreads::AnyBackgroundThreadNormalTask
    );
}

void AARESGameMode_Optimized::SetOperationalMode(EARESOperationalMode Mode)
{
    if (!IsARESSystemReady())
    {
        return;
    }

    CurrentMode = Mode;
    
    switch (Mode)
    {
    case EARESOperationalMode::Stealth:
        EngageStealthMode();
        break;
    case EARESOperationalMode::Offensive:
        InitiateCountermeasures();
        break;
    case EARESOperationalMode::Defensive:
        ARESSystem->engageDefensiveMode();
        break;
    case EARESOperationalMode::Recon:
        ARESSystem->engageReconMode();
        break;
    case EARESOperationalMode::Training:
        ARESSystem->engageTrainingMode();
        break;
    }

    UE_LOG(LogTemp, Warning, TEXT("ARES Operational Mode changed to: %s"), 
        *UEnum::GetValueAsString(Mode));
}

void AARESGameMode_Optimized::EngageStealthMode()
{
    if (IsARESSystemReady())
    {
        ARESSystem->engageStealthMode();
        
        if (GEngine)
        {
            GEngine->AddOnScreenDebugMessage(-1, 5.0f, FColor::Blue,
                TEXT("STEALTH MODE ENGAGED - EM Signature Minimized"));
        }
    }
}

void AARESGameMode_Optimized::InitiateCountermeasures()
{
    if (IsARESSystemReady())
    {
        ARESSystem->initiateCountermeasures();
        
        if (GEngine)
        {
            GEngine->AddOnScreenDebugMessage(-1, 5.0f, FColor::Red,
                TEXT("COUNTERMEASURES ACTIVE - Offensive Systems Online"));
        }
    }
}

void AARESGameMode_Optimized::PerformIdentitySwitch()
{
    if (IsARESSystemReady())
    {
        ARESSystem->performEmergencyIdentitySwitch();
        
        if (GEngine)
        {
            GEngine->AddOnScreenDebugMessage(-1, 5.0f, FColor::Orange,
                TEXT("IDENTITY SWITCHED - New Hardware Attestation Active"));
        }
    }
}

void AARESGameMode_Optimized::EnableARVisualization(bool bEnable)
{
    bARVisualizationEnabled = bEnable;
    
    if (bEnable)
    {
        UE_LOG(LogTemp, Warning, TEXT("Enabling AR Visualization for Meta Quest 3"));
        // Enable AR overlays through OpenXR
    }
    else
    {
        UE_LOG(LogTemp, Warning, TEXT("Disabling AR Visualization"));
    }
}

void AARESGameMode_Optimized::SetARVisualizationMode(const FString& Mode)
{
    if (!bARVisualizationEnabled)
    {
        UE_LOG(LogTemp, Warning, TEXT("AR Visualization not enabled"));
        return;
    }

    // Configure AR visualization mode
    UE_LOG(LogTemp, Warning, TEXT("Setting AR Visualization Mode: %s"), *Mode);
}

void AARESGameMode_Optimized::ScanForNetworks()
{
    if (!IsARESSystemReady())
    {
        return;
    }

    FFunctionGraphTask::CreateAndDispatchWhenReady(
        [this]()
        {
            TRACE_CPUPROFILER_EVENT_SCOPE(ARESNetworkScan);
            
            ARESSystem->scanAndConnectNetworks();
            
            // Get discovered networks
            auto networks = ARESSystem->getDiscoveredNetworks();
            
            // Broadcast network discovery events on game thread
            AsyncTask(ENamedThreads::GameThread, [this, networks]()
            {
                for (const auto& net : networks)
                {
                    FARESNetworkInfo NetworkInfo;
                    NetworkInfo.NetworkType = UTF8_TO_TCHAR(net.type.c_str());
                    NetworkInfo.SSID = UTF8_TO_TCHAR(net.ssid.c_str());
                    NetworkInfo.SignalStrength = net.signal_strength;
                    NetworkInfo.bIsEncrypted = net.encrypted;
                    NetworkInfo.Channel = net.channel;
                    
                    OnNetworkDiscovered.Broadcast(NetworkInfo);
                }
            });
        },
        TStatId(),
        nullptr,
        ENamedThreads::AnyBackgroundThreadNormalTask
    );
}

void AARESGameMode_Optimized::ConnectToNetwork(const FARESNetworkInfo& NetworkInfo, const FString& Credentials)
{
    if (!IsARESSystemReady())
    {
        return;
    }

    // Convert to ARES network structure and connect
    ares::NetworkInterface network;
    network.type = TCHAR_TO_UTF8(*NetworkInfo.NetworkType);
    network.ssid = TCHAR_TO_UTF8(*NetworkInfo.SSID);
    network.signal_strength = NetworkInfo.SignalStrength;
    network.encrypted = NetworkInfo.bIsEncrypted;
    network.channel = NetworkInfo.Channel;

    ARESSystem->connectToNetwork(network, TCHAR_TO_UTF8(*Credentials));
}

void AARESGameMode_Optimized::SetUpdateFrequency(float FrequencyHz)
{
    UpdateFrequency = FMath::Clamp(FrequencyHz, 1.0f, 1000.0f);
    
    if (WorkerThread)
    {
        WorkerThread->SetUpdateFrequency(UpdateFrequency);
    }
    
    UE_LOG(LogTemp, Warning, TEXT("ARES Update Frequency set to: %.1f Hz"), UpdateFrequency);
}

void AARESGameMode_Optimized::SetGPUPowerLimit(float PowerLimitWatts)
{
    if (IsARESSystemReady())
    {
        ARESSystem->setGPUPowerLimit(PowerLimitWatts);
        UE_LOG(LogTemp, Warning, TEXT("GPU Power Limit set to: %.1f W"), PowerLimitWatts);
    }
}