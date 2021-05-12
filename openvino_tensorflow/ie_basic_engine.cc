/*******************************************************************************
 * Copyright (C) 2021 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 *******************************************************************************/

#include <iostream>

#include "logging/ovtf_log.h"
#include "openvino_tensorflow/ie_basic_engine.h"
#include "openvino_tensorflow/ie_utils.h"

using namespace InferenceEngine;

namespace tensorflow {
namespace openvino_tensorflow {

IE_Basic_Engine::IE_Basic_Engine(InferenceEngine::CNNNetwork ie_network,
                                 std::string device)
    : IE_Backend_Engine(ie_network, device) {}

IE_Basic_Engine::~IE_Basic_Engine() {}

void IE_Basic_Engine::infer(
    std::vector<std::shared_ptr<IETensor>>& inputs,
    std::vector<std::string>& input_names,
    std::vector<std::shared_ptr<IETensor>>& outputs,
    std::vector<std::string>& output_names,
    std::vector<std::shared_ptr<IETensor>>& hoisted_params,
    std::vector<std::string>& param_names) {
  load_network();
  if (m_infer_reqs.empty()) {
    m_infer_reqs.push_back(m_exe_network.CreateInferRequest());
  }

  //  Prepare input blobs
  auto func = m_network.getFunction();
  auto parameters = func->get_parameters();
  for (int i = 0; i < inputs.size(); i++) {
    if (inputs[i] != nullptr) {
      auto input_blob = m_infer_reqs[0].GetBlob(input_names[i]);
      MemoryBlob::Ptr minput = as<MemoryBlob>(input_blob);
      auto minputHolder = minput->wmap();

      auto inputBlobData = minputHolder.as<uint8_t*>();
      size_t input_data_size = input_blob->byteSize();
      //inputs[i]->read((void*)inputBlobData, input_data_size);
      std::cout << "LOG - input " << i << " - type: " << input_blob->getTensorDesc().getPrecision() << std::endl;
    }
  }
  for (int i = 0; i < inputs.size(); i++) {
    if (inputs[i] != nullptr) {
#if defined(OPENVINO_2021_2)
      if (m_device != "MYRIAD" && m_device != "HDDL")
        m_infer_reqs[0].SetBlob(input_names[i], inputs[i]->get_blob());
      else {
        auto input_blob = m_infer_reqs[0].GetBlob(input_names[i]);
        MemoryBlob::Ptr minput = as<MemoryBlob>(input_blob);
        auto minputHolder = minput->wmap();

        auto inputBlobData = minputHolder.as<uint8_t*>();
        size_t input_data_size = input_blob->byteSize();
        inputs[i]->read((void*)inputBlobData, input_data_size);
      }
#else
      m_infer_reqs[0].SetBlob(input_names[i], inputs[i]->get_blob());
#endif
    }
  }
  for (int i = 0; i < inputs.size(); i++) {
    if (inputs[i] != nullptr) {
      auto input_blob = m_infer_reqs[0].GetBlob(input_names[i]);
      MemoryBlob::Ptr minput = as<MemoryBlob>(input_blob);
      auto minputHolder = minput->wmap();

      auto inputBlobData = minputHolder.as<uint8_t*>();
      size_t input_data_size = input_blob->byteSize();
      //inputs[i]->read((void*)inputBlobData, input_data_size);
      std::cout << "LOG - input " << i;
      uint64_t sum = 0;
      for (int j=0; j<input_blob->byteSize(); j++) {
        sum += (uint64_t)(inputBlobData[j]);
      }
      std::cout << " - sum: " << sum << std::endl;
    }
  }

  for (int i = 0; i < hoisted_params.size(); i++) {
    if (hoisted_params[i] != nullptr)
      m_infer_reqs[0].SetBlob(param_names[i], hoisted_params[i]->get_blob());
  }

  //  Prepare output blobs
  auto results = func->get_results();
  for (int i = 0; i < results.size(); i++) {
    if (outputs[i] != nullptr) {
      OVTF_VLOG(4) << "Executable::call() SetBlob()";
      m_infer_reqs[0].SetBlob(output_names[i], outputs[i]->get_blob());
    }
  }

  m_infer_reqs[0].Infer();

  // Set dynamic output blobs
  for (int i = 0; i < results.size(); i++) {
    if (outputs[i] == nullptr) {
      OVTF_VLOG(4) << "Executable::call() GetBlob()";
      auto blob = m_infer_reqs[0].GetBlob(output_names[i]);
      outputs[i] = std::make_shared<IETensor>(blob);
    }
  }
  OVTF_VLOG(4) << "Inference Successful";
  for (int i = 0; i < outputs.size(); i++) {
    if (outputs[i] != nullptr) {
      auto output_blob = m_infer_reqs[0].GetBlob(output_names[i]);
      MemoryBlob::Ptr moutput = as<MemoryBlob>(output_blob);
      auto moutputHolder = moutput->wmap();

      auto outputBlobData = moutputHolder.as<uint8_t*>();
      size_t output_data_size = output_blob->byteSize();
      //inputs[i]->read((void*)inputBlobData, input_data_size);
      std::cout << "LOG - output " << i << " - type: " << output_blob->getTensorDesc().getPrecision() << " - size: " << output_data_size << " - data: ";
      float sum = 0;
      for (int j=0; j<output_blob->byteSize()/sizeof(float); j++) {
        sum += (float)(((float*)outputBlobData)[j]);
	if (j < 5)
	  std::cout << (float)(((float*)outputBlobData)[j]) << ", ";
      }
      std::cout << " - sum: " << sum << std::endl;
    }
  }

  // return true;
}
}  // namespace openvino_tensorflow
}  // namespace tensorflow
