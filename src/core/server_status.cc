// Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "src/core/server_status.h"

#include <time.h>
#include "src/core/constants.h"
#include "src/core/logging.h"
#include "src/core/metric_model_reporter.h"
#include "src/core/metrics.h"
#include "src/core/tracing.h"

namespace nvidia { namespace inferenceserver {

Status
ServerStatusManager::InitForModel(
    const std::string& model_name, const ModelConfig& model_config)
{
  std::lock_guard<std::mutex> lock(mu_);

  auto& ms = *server_status_.mutable_model_status();
  if (ms.find(model_name) == ms.end()) {
    LOG_INFO << "New status tracking for model '" << model_name << "'";
  } else {
    LOG_INFO << "New status tracking for re-added model '" << model_name << "'";
    ms[model_name].Clear();
  }

  ms[model_name].mutable_config()->CopyFrom(model_config);

  return Status::Success;
}

Status
ServerStatusManager::UpdateConfigForModel(
    const std::string& model_name, const ModelConfig& model_config)
{
  std::lock_guard<std::mutex> lock(mu_);

  auto& ms = *server_status_.mutable_model_status();
  if (ms.find(model_name) == ms.end()) {
    return Status(
        Status::Code::INVALID_ARG,
        "try to update config for non-existing model '" + model_name + "'");
  } else {
    LOG_INFO << "Updating config for model '" << model_name << "'";
  }

  ms[model_name].mutable_config()->CopyFrom(model_config);

  return Status::Success;
}

Status
ServerStatusManager::SetModelVersionReadyState(
    const std::string& model_name, int64_t version, ModelReadyState state,
    const ModelReadyStateReason& state_reason)
{
  std::lock_guard<std::mutex> lock(mu_);
  auto itr = server_status_.mutable_model_status()->find(model_name);
  if (itr == server_status_.model_status().end()) {
    return Status(
        Status::Code::INVALID_ARG,
        "fail to update ready state for unknown model '" + model_name + "'");
  }

  auto vitr = itr->second.mutable_version_status()->find(version);
  if (vitr == itr->second.version_status().end()) {
    // Completely fresh
    ModelVersionStatus version_status;
    version_status.set_ready_state(state);
    *version_status.mutable_ready_state_reason() = state_reason;
    (*(itr->second.mutable_version_status()))[version] = version_status;
  } else {
    vitr->second.set_ready_state(state);
    *(vitr->second.mutable_ready_state_reason()) = state_reason;
  }

  return Status::Success;
}

Status
ServerStatusManager::Get(ServerStatus* server_status) const
{
  std::lock_guard<std::mutex> lock(mu_);
  server_status->CopyFrom(server_status_);

  return Status::Success;
}

Status
ServerStatusManager::Get(
    ServerStatus* server_status, const std::string& model_name) const
{
  std::lock_guard<std::mutex> lock(mu_);

  server_status->Clear();

  const auto& itr = server_status_.model_status().find(model_name);
  if (itr == server_status_.model_status().end()) {
    return Status(
        Status::Code::INVALID_ARG,
        "no status available for unknown model '" + model_name + "'");
  }

  auto& ms = *server_status->mutable_model_status();
  ms[model_name].CopyFrom(itr->second);

  return Status::Success;
}

void
ServerStatusManager::UpdateFailedInferStats(
    const std::string& model_name, const int64_t model_version,
    uint64_t last_timestamp_ms, uint64_t request_duration_ns)
{
  std::lock_guard<std::mutex> lock(mu_);

  // Model must exist...
  auto itr = server_status_.mutable_model_status()->find(model_name);
  if (itr == server_status_.model_status().end()) {
    LOG_ERROR << "can't update INFER duration stat for " << model_name;
  } else {
    // batch_size may be zero if the failure occurred before it could
    // be determined... but we still record the failure.

    // model version
    auto& mvs = *itr->second.mutable_version_status();
    auto mvs_itr = mvs.find(model_version);
    if (mvs_itr == mvs.end()) {
      ModelVersionStatus& version_status = mvs[model_version];
      version_status.set_last_inference_timestamp_milliseconds(
          last_timestamp_ms);
      auto stats = version_status.mutable_infer_stats();
      stats->mutable_failed()->set_count(1);
      stats->mutable_failed()->set_total_time_ns(request_duration_ns);
    } else {
      ModelVersionStatus& version_status = mvs_itr->second;
      if (last_timestamp_ms > 0) {
        version_status.set_last_inference_timestamp_milliseconds(
            last_timestamp_ms);
      }
      auto stats = version_status.mutable_infer_stats();
      stats->mutable_failed()->set_count(stats->failed().count() + 1);
      stats->mutable_failed()->set_total_time_ns(
          stats->failed().total_time_ns() + request_duration_ns);
    }
  }
}

void
ServerStatusManager::UpdateSuccessInferStats(
    const std::string& model_name, const int64_t model_version,
    uint64_t last_timestamp_ms, uint64_t request_duration_ns,
    uint64_t queue_duration_ns)
{
  std::lock_guard<std::mutex> lock(mu_);

  // Model must exist...
  auto itr = server_status_.mutable_model_status()->find(model_name);
  if (itr == server_status_.model_status().end()) {
    LOG_ERROR << "can't update duration stat for " << model_name;
  } else {
    // model version
    auto& mvs = *itr->second.mutable_version_status();
    auto mvs_itr = mvs.find(model_version);
    InferRequestStats* stats = nullptr;
    if (mvs_itr == mvs.end()) {
      ModelVersionStatus& version_status = mvs[model_version];
      version_status.set_last_inference_timestamp_milliseconds(
          last_timestamp_ms);
      stats = version_status.mutable_infer_stats();
    } else {
      ModelVersionStatus& version_status = mvs_itr->second;
      if (last_timestamp_ms > 0) {
        version_status.set_last_inference_timestamp_milliseconds(
            last_timestamp_ms);
      }

      stats = version_status.mutable_infer_stats();
    }

    if (stats != nullptr) {
      stats->mutable_success()->set_count(stats->success().count() + 1);
      stats->mutable_success()->set_total_time_ns(
          stats->success().total_time_ns() + request_duration_ns);
      stats->mutable_queue()->set_count(stats->queue().count() + 1);
      stats->mutable_queue()->set_total_time_ns(
          stats->queue().total_time_ns() + queue_duration_ns);
    } else {
      LOG_ERROR << "Internal error logging INFER stats for " << model_name;
    }
  }
}

void
ServerStatusManager::UpdateSuccessInferBatchStats(
    const std::string& model_name, const int64_t model_version,
    size_t batch_size, size_t success_request_count,
    uint64_t compute_input_duration_ns, uint64_t compute_infer_duration_ns,
    uint64_t compute_output_duration_ns)
{
  std::lock_guard<std::mutex> lock(mu_);

  // Model must exist...
  auto itr = server_status_.mutable_model_status()->find(model_name);
  if (itr == server_status_.model_status().end()) {
    LOG_ERROR << "can't update duration stat for " << model_name;
  } else {
    // model version
    auto& mvs = *itr->second.mutable_version_status();
    auto mvs_itr = mvs.find(model_version);
    InferRequestStats* stats = nullptr;
    InferBatchStats* batch_stats = nullptr;
    if (mvs_itr == mvs.end()) {
      ModelVersionStatus& version_status = mvs[model_version];
      stats = version_status.mutable_infer_stats();
      batch_stats =
          &((*version_status.mutable_infer_batch_stats())[batch_size]);
    } else {
      ModelVersionStatus& version_status = mvs_itr->second;
      stats = version_status.mutable_infer_stats();
      batch_stats =
          &((*version_status.mutable_infer_batch_stats())[batch_size]);
    }

    if (stats != nullptr) {
      stats->mutable_compute_input()->set_count(
          stats->compute_input().count() + success_request_count);
      stats->mutable_compute_input()->set_total_time_ns(
          stats->compute_input().total_time_ns() +
          compute_input_duration_ns * success_request_count);
      stats->mutable_compute_infer()->set_count(
          stats->compute_infer().count() + success_request_count);
      stats->mutable_compute_infer()->set_total_time_ns(
          stats->compute_infer().total_time_ns() +
          compute_infer_duration_ns * success_request_count);
      stats->mutable_compute_output()->set_count(
          stats->compute_output().count() + success_request_count);
      stats->mutable_compute_output()->set_total_time_ns(
          stats->compute_output().total_time_ns() +
          compute_output_duration_ns * success_request_count);
    } else {
      LOG_ERROR << "Internal error logging INFER stats for " << model_name;
    }

    if (batch_stats != nullptr) {
      batch_stats->mutable_compute_input()->set_count(
          batch_stats->compute_input().count() + 1);
      batch_stats->mutable_compute_input()->set_total_time_ns(
          batch_stats->compute_input().total_time_ns() +
          compute_input_duration_ns);
      batch_stats->mutable_compute_infer()->set_count(
          batch_stats->compute_infer().count() + 1);
      batch_stats->mutable_compute_infer()->set_total_time_ns(
          batch_stats->compute_infer().total_time_ns() +
          compute_infer_duration_ns);
      batch_stats->mutable_compute_output()->set_count(
          batch_stats->compute_output().count() + 1);
      batch_stats->mutable_compute_output()->set_total_time_ns(
          batch_stats->compute_output().total_time_ns() +
          compute_output_duration_ns);
    } else {
      LOG_ERROR << "Internal error logging INFER batch stats for "
                << model_name;
    }
  }
}

#ifdef TRTIS_ENABLE_STATS

void
ModelInferStats::NewTrace(Trace* parent)
{
#ifdef TRTIS_ENABLE_TRACING
  if (trace_manager_ != nullptr) {
    auto ltrace_manager = reinterpret_cast<OpaqueTraceManager*>(trace_manager_);
    trace_ = nullptr;
    if (trace_manager_->using_triton_) {
      ltrace_manager->triton_create_fn_(
          reinterpret_cast<TRITONSERVER_Trace**>(&trace_), model_name_.c_str(),
          requested_model_version_, ltrace_manager->userp_);
    } else {
      ltrace_manager->create_fn_(
          reinterpret_cast<TRITONSERVER_Trace**>(&trace_), model_name_.c_str(),
          requested_model_version_, ltrace_manager->userp_);
    }
    if (trace_ != nullptr) {
      trace_->SetModelName(model_name_);
      trace_->SetModelVersion(requested_model_version_);
      if (parent != nullptr) {
        trace_->SetParentId(parent->Id());
      }
    }
  }
#endif  // TRTIS_ENABLE_TRACING
}

void
ModelInferStats::Report()
{
#ifdef TRTIS_ENABLE_TRACING
  if (trace_ != nullptr) {
    trace_->Report(this);
    // Inform that the trace object is done and can be released
    if (trace_manager_->using_triton_) {
      trace_manager_->triton_release_fn_(
          reinterpret_cast<TRITONSERVER_Trace*>(trace_),
          trace_->ActivityUserp(), trace_manager_->userp_);
    } else {
      trace_manager_->release_fn_(
          reinterpret_cast<TRITONSERVER_Trace*>(trace_),
          trace_->ActivityUserp(), trace_manager_->userp_);
    }
  }
#endif  // TRTIS_ENABLE_TRACING

  // If the inference request failed before a backend could be
  // determined, there will be no metrics reporter.. so just use the
  // version directly from the inference request.
  const int64_t model_version = (metric_reporter_ != nullptr)
                                    ? metric_reporter_->ModelVersion()
                                    : requested_model_version_;

  const uint64_t request_duration_ns =
      Duration(TimestampKind::kRequestStart, TimestampKind::kRequestEnd);
  struct timespec last_ts;
  clock_gettime(CLOCK_REALTIME, &last_ts);
  const uint64_t last_timestamp_ms = TIMESPEC_TO_MILLIS(last_ts);

  if (failed_) {
    status_manager_->UpdateFailedInferStats(
        model_name_, model_version, last_timestamp_ms, request_duration_ns);
#ifdef TRTIS_ENABLE_METRICS
    if (metric_reporter_ != nullptr) {
      metric_reporter_->MetricInferenceFailure(gpu_device_).Increment();
    }
#endif  // TRTIS_ENABLE_METRICS
  } else {
    uint64_t queue_duration_ns =
        extra_queue_duration_ +
        Duration(TimestampKind::kQueueStart, TimestampKind::kComputeStart);
    uint64_t compute_duration_ns =
        extra_compute_duration_ +
        Duration(TimestampKind::kComputeStart, TimestampKind::kComputeEnd);

// FIXME should be done in BackendContext::Run()
#if 0
    uint64_t compute_input_duration_ns =
        extra_compute_input_duration_ +
        Duration(TimestampKind::kComputeStart, TimestampKind::kComputeInputEnd);
    uint64_t compute_infer_duration_ns =
        extra_compute_infer_duration_ + Duration(
                                            TimestampKind::kComputeInputEnd,
                                            TimestampKind::kComputeOutputStart);
    uint64_t compute_output_duration_ns =
        extra_compute_output_duration_ +
        Duration(
            TimestampKind::kComputeOutputStart, TimestampKind::kComputeEnd);

#endif
    status_manager_->UpdateSuccessInferStats(
        model_name_, model_version, last_timestamp_ms, request_duration_ns,
        queue_duration_ns);

#ifdef TRTIS_ENABLE_METRICS
    if (metric_reporter_ != nullptr) {
      metric_reporter_->MetricInferenceSuccess(gpu_device_).Increment();
      metric_reporter_->MetricInferenceCount(gpu_device_)
          .Increment(batch_size_);
      if (execution_count_ > 0) {
        metric_reporter_->MetricInferenceExecutionCount(gpu_device_)
            .Increment(execution_count_);
      }

      metric_reporter_->MetricInferenceRequestDuration(gpu_device_)
          .Increment(request_duration_ns / 1000);
      metric_reporter_->MetricInferenceComputeDuration(gpu_device_)
          .Increment(compute_duration_ns / 1000);
      metric_reporter_->MetricInferenceQueueDuration(gpu_device_)
          .Increment(queue_duration_ns / 1000);

      metric_reporter_->MetricInferenceLoadRatio(gpu_device_)
          .Observe(
              (double)request_duration_ns /
              std::max(1.0, (double)compute_duration_ns));
    }
#endif  // TRTIS_ENABLE_METRICS
  }
}

void
ModelInferStats::IncrementQueueDuration(const ModelInferStats& other)
{
  extra_queue_duration_ +=
      other.Duration(TimestampKind::kQueueStart, TimestampKind::kComputeStart);
}

void
ModelInferStats::IncrementComputeDuration(const ModelInferStats& other)
{
  extra_compute_duration_ +=
      other.Duration(TimestampKind::kComputeStart, TimestampKind::kComputeEnd);
}

uint64_t
ModelInferStats::Duration(
    ModelInferStats::TimestampKind start_kind,
    ModelInferStats::TimestampKind end_kind) const
{
  const struct timespec& start = Timestamp(start_kind);
  const struct timespec& end = Timestamp(end_kind);
  uint64_t start_ns = TIMESPEC_TO_NANOS(start);
  uint64_t end_ns = TIMESPEC_TO_NANOS(end);

  // If the start or end timestamp is 0 then can't calculate the
  // duration, so return 0.
  if ((start_ns == 0) || (end_ns == 0)) {
    return 0;
  }

  return (start_ns > end_ns) ? 0 : end_ns - start_ns;
}

#endif  // TRTIS_ENABLE_STATS

}}  // namespace nvidia::inferenceserver
