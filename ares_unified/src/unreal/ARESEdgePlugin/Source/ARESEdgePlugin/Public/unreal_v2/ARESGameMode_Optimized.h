// PROPRIETARY AND CONFIDENTIAL
// Copyright (c) 2024 DELFICTUS I/O LLC
// Patent Pending - Application #63/826,067

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/GameModeBase.h"
#include "Engine/World.h"
#include "HAL/ThreadSafeBool.h"
#include "Containers/CircularQueue.h"
#include "ARESGameMode_Optimized.generated.h"

// Forward declarations
namespace ares {
    class UnifiedQuantumARES;
    namespace chronopath {
        class DRPPChronopathEngine;
    }
}

UENUM(BlueprintType)
enum class EARESOperationalMode : uint8
{
    Stealth         UMETA(DisplayName = "Stealth Mode"),
    Offensive       UMETA(DisplayName = "Offensive Mode"),
    Defensive       UMETA(DisplayName = "Defensive Mode"),
    Recon           UMETA(DisplayName = "Reconnaissance"),
    Training        UMETA(DisplayName = "Training Mode")
};

USTRUCT(BlueprintType)
struct FARESNetworkInfo
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly)
    FString NetworkType;

    UPROPERTY(BlueprintReadOnly)
    FString SSID;

    UPROPERTY(BlueprintReadOnly)
    float SignalStrength;

    UPROPERTY(BlueprintReadOnly)
    bool bIsEncrypted;

    UPROPERTY(BlueprintReadOnly)
    int32 Channel;
};

USTRUCT(BlueprintType)
struct FARESSystemMetrics
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly)
    float EnergyLevel = 1.0f;

    UPROPERTY(BlueprintReadOnly)
    int32 ThreatsDetected = 0;

    UPROPERTY(BlueprintReadOnly)
    float StealthEffectiveness = 1.0f;

    UPROPERTY(BlueprintReadOnly)
    float GPUMemoryUsageMB = 0.0f;

    UPROPERTY(BlueprintReadOnly)
    float CPUUsagePercent = 0.0f;

    UPROPERTY(BlueprintReadOnly)
    int32 ActiveSwarmNodes = 0;

    UPROPERTY(BlueprintReadOnly)
    float ConsensusHealth = 1.0f;
};

DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnARESMetricsUpdated, const FARESSystemMetrics&, Metrics);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnARESAIResponse, const FString&, Response);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnARESNetworkDiscovered, const FARESNetworkInfo&, NetworkInfo);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_TwoParams(FOnARESThreatDetected, int32, ThreatID, float, ThreatLevel);

UCLASS()
class ARESEDGEPLUGIN_API AARESGameMode_Optimized : public AGameModeBase
{
    GENERATED_BODY()

public:
    AARESGameMode_Optimized();
    virtual ~AARESGameMode_Optimized();

    // UE5 Lifecycle
    virtual void InitGame(const FString& MapName, const FString& Options, FString& ErrorMessage) override;
    virtual void StartPlay() override;
    virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;
    virtual void Tick(float DeltaSeconds) override;

    // ARES Core Functions
    UFUNCTION(BlueprintCallable, Category = "ARES", meta = (CallInEditor = "true"))
    void InitializeARESSystem();

    UFUNCTION(BlueprintCallable, Category = "ARES")
    void ShutdownARESSystem();

    UFUNCTION(BlueprintCallable, Category = "ARES")
    bool IsARESSystemReady() const;

    UFUNCTION(BlueprintCallable, Category = "ARES")
    void ConfigureAIProvider(const FString& Provider, const FString& APIKey);

    UFUNCTION(BlueprintCallable, Category = "ARES|AI")
    void QueryAI(const FString& Prompt, const FString& Strategy = "consensus");

    UFUNCTION(BlueprintCallable, Category = "ARES|Operations")
    void SetOperationalMode(EARESOperationalMode Mode);

    UFUNCTION(BlueprintPure, Category = "ARES|Operations")
    EARESOperationalMode GetOperationalMode() const { return CurrentMode; }

    UFUNCTION(BlueprintCallable, Category = "ARES|Stealth")
    void EngageStealthMode();

    UFUNCTION(BlueprintCallable, Category = "ARES|Combat")
    void InitiateCountermeasures();

    UFUNCTION(BlueprintCallable, Category = "ARES|Identity")
    void PerformIdentitySwitch();

    // Real-time metrics
    UFUNCTION(BlueprintPure, Category = "ARES|Metrics")
    FARESSystemMetrics GetSystemMetrics() const { return CachedMetrics; }

    // AR/VR Support
    UFUNCTION(BlueprintCallable, Category = "ARES|XR")
    void EnableARVisualization(bool bEnable);

    UFUNCTION(BlueprintCallable, Category = "ARES|XR")
    void SetARVisualizationMode(const FString& Mode);

    // Network discovery
    UFUNCTION(BlueprintCallable, Category = "ARES|Network")
    void ScanForNetworks();

    UFUNCTION(BlueprintCallable, Category = "ARES|Network")
    void ConnectToNetwork(const FARESNetworkInfo& NetworkInfo, const FString& Credentials = "");

    // Performance controls
    UFUNCTION(BlueprintCallable, Category = "ARES|Performance")
    void SetUpdateFrequency(float FrequencyHz);

    UFUNCTION(BlueprintCallable, Category = "ARES|Performance")
    void SetGPUPowerLimit(float PowerLimitWatts);

    // Delegates
    UPROPERTY(BlueprintAssignable, Category = "ARES|Events")
    FOnARESMetricsUpdated OnMetricsUpdated;

    UPROPERTY(BlueprintAssignable, Category = "ARES|Events")
    FOnARESAIResponse OnAIResponseReceived;

    UPROPERTY(BlueprintAssignable, Category = "ARES|Events")
    FOnARESNetworkDiscovered OnNetworkDiscovered;

    UPROPERTY(BlueprintAssignable, Category = "ARES|Events")
    FOnARESThreatDetected OnThreatDetected;

protected:
    // Thread-safe message queue for AI responses
    struct FAIMessage
    {
        FString Response;
        float Timestamp;
    };

    TCircularQueue<FAIMessage> AIResponseQueue;
    FCriticalSection AIQueueMutex;

    // Metrics update
    void UpdateMetrics();
    void ProcessPendingAIResponses();

    // ARES System pointer
    TUniquePtr<ares::UnifiedQuantumARES> ARESSystem;

    // Cached metrics for thread-safe access
    FARESSystemMetrics CachedMetrics;
    mutable FCriticalSection MetricsMutex;

    // Status
    UPROPERTY(BlueprintReadOnly, Category = "ARES|Status")
    EARESOperationalMode CurrentMode = EARESOperationalMode::Stealth;

    UPROPERTY(BlueprintReadOnly, Category = "ARES|Status")
    bool bSystemInitialized = false;

    UPROPERTY(BlueprintReadOnly, Category = "ARES|Status")
    bool bARVisualizationEnabled = false;

private:
    // Background threads
    class FARESWorkerThread* WorkerThread = nullptr;
    FRunnableThread* WorkerThreadHandle = nullptr;
    FThreadSafeBool bShutdownRequested;

    // Performance tracking
    float UpdateFrequency = 100.0f; // Hz
    float MetricsUpdateInterval = 0.1f; // seconds
    float LastMetricsUpdateTime = 0.0f;

    // Frame timing
    double LastTickTime;
    double AccumulatedTickTime;
    int32 TickCount;

    // Performance stats
    struct FPerformanceStats
    {
        float AverageFrameTimeMs = 0.0f;
        float MaxFrameTimeMs = 0.0f;
        float MinFrameTimeMs = 999.0f;
        int32 FrameCount = 0;
        
        void Reset()
        {
            AverageFrameTimeMs = 0.0f;
            MaxFrameTimeMs = 0.0f;
            MinFrameTimeMs = 999.0f;
            FrameCount = 0;
        }
    };
    
    FPerformanceStats PerfStats;
    FCriticalSection PerfStatsMutex;

    // Helper functions
    void InitializeWorkerThread();
    void ShutdownWorkerThread();
    void UpdatePerformanceStats(float DeltaTime);
};