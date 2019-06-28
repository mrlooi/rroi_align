#include "cpu/vision.h"
#include "rotate_rect_ops.h"

#include <iostream>

template <typename scalar_t>
at::Tensor rotate_nms_cpu_kernel(const at::Tensor& dets,
                          const at::Tensor& scores,
                          const float threshold) {
  AT_ASSERTM(!dets.type().is_cuda(), "dets must be a CPU tensor");
  AT_ASSERTM(!scores.type().is_cuda(), "scores must be a CPU tensor");
  AT_ASSERTM(dets.type() == scores.type(), "dets should have the same type as scores");

  if (dets.numel() == 0) {
    return at::empty({0}, dets.options().dtype(at::kLong).device(at::kCPU));
  }

  auto xc_t = dets.select(1, 0).contiguous();
  auto yc_t = dets.select(1, 1).contiguous();
  auto w_t = dets.select(1, 2).contiguous();
  auto h_t = dets.select(1, 3).contiguous();
  auto angle_t = dets.select(1, 4).contiguous();

  at::Tensor areas_t = w_t * h_t;

  auto order_t = std::get<1>(scores.sort(0, /* descending=*/true));

  auto ndets = dets.size(0);
  at::Tensor suppressed_t = at::zeros({ndets}, dets.options().dtype(at::kByte).device(at::kCPU));

  auto suppressed = suppressed_t.data<uint8_t>();
  auto order = order_t.data<int64_t>();
  auto xc = xc_t.data<scalar_t>();
  auto yc = yc_t.data<scalar_t>();
  auto w = w_t.data<scalar_t>();
  auto h = h_t.data<scalar_t>();
  auto angle = angle_t.data<scalar_t>();
  auto areas = areas_t.data<scalar_t>();

  scalar_t rect_1[5];
  for (int64_t _i = 0; _i < ndets; _i++) {
    auto i = order[_i];
    if (suppressed[i] == 1)
      continue;

    rect_1[0] = xc[i];
    rect_1[1] = yc[i];
    rect_1[2] = w[i];
    rect_1[3] = h[i];
    rect_1[4] = angle[i];

    auto iarea = areas[i];

    scalar_t rect_2[5];
    for (int64_t _j = _i + 1; _j < ndets; _j++) {
      auto j = order[_j];
      if (suppressed[j] == 1)
        continue;

      rect_2[0] = xc[j];
      rect_2[1] = yc[j];
      rect_2[2] = w[j];
      rect_2[3] = h[j];
      rect_2[4] = angle[j];
      
      auto inter_area = inter(rect_1, rect_2);
      auto ovr = inter_area / (iarea + areas[j] - inter_area);
      if (ovr >= threshold)
        suppressed[j] = 1;
   }
  }
  return at::nonzero(suppressed_t == 0).squeeze(1);
}


template <typename scalar_t>
at::Tensor rotate_soft_nms_cpu_kernel(at::Tensor& dets,
                          at::Tensor& scores,
                          at::Tensor& indices,
                          const float nms_thresh,
                          const float sigma,
                          const float score_thresh,
                          const int method
                          ) 
{
  AT_ASSERTM(!dets.type().is_cuda(), "dets must be a CPU tensor");
  AT_ASSERTM(!scores.type().is_cuda(), "scores must be a CPU tensor");
  AT_ASSERTM(dets.type() == scores.type(), "dets should have the same type as scores");

  if (dets.numel() == 0) {
    return at::empty({0}, dets.options().dtype(at::kLong).device(at::kCPU));
  }

  auto ndets = dets.size(0);
  auto scores_d = scores.contiguous().data<scalar_t>();
  auto dets_d = dets.contiguous().data<scalar_t>();
  
  auto indices_d = indices.contiguous().data<int64_t>();

  for (int64_t i = 0; i < ndets; i++) {
    int64_t pos = i + 1;
    auto rem_scores = scores.narrow(/*dim=*/0, /*start=*/pos, /*length=*/ndets - pos);
    // auto order_t = std::get<1>(rem_scores.sort(0, /* descending=*/true));
    // auto order = order_t.data<int64_t>();
    int64_t maxpos = 0;
    scalar_t maxscore = scores_d[ndets - 1];
    if (i != ndets - 1)
    {
      maxpos = rem_scores.argmax().data<int64_t>()[0];
      maxscore = rem_scores.data<scalar_t>()[maxpos];
    }
    if (scores_d[i] < maxscore)
    {
      scalar_t* dd = dets_d + (i*5);
      scalar_t* dd_max = dets_d + ((maxpos+pos)*5);
      scalar_t tmp;
      for (size_t n = 0; n < 5; n++)
      {
        tmp = dd[n];
        dd[n] = dd_max[n];
        dd_max[n] = tmp;
      }
      tmp = scores_d[i];
      scores_d[i] = maxscore;
      scores_d[maxpos+pos] = tmp;

      int64_t tmp_i = indices_d[i];
      indices_d[i] = indices_d[maxpos + pos];
      indices_d[maxpos + pos] = tmp_i;

    }

    const scalar_t* bbox = dets_d + (i*5);
    auto iw = bbox[2];
    auto ih = bbox[3];
    auto iarea = iw * ih;

    for (int64_t j = pos; j < ndets; j++) {

      const scalar_t* bbox2 = dets_d + (j*5);

      auto jw = bbox2[2];
      auto jh = bbox2[3];
      auto jarea = jw * jh;
      auto inter_area = inter(bbox, bbox2);
      auto ovr = inter_area / (iarea + jarea - inter_area);

      float weight = 1.0f;
      if (method == NMS_METHOD::GAUSSIAN)
      {
        weight = exp(-(ovr * ovr) / sigma);
      } else if (ovr >= nms_thresh)
      {
        weight = method == NMS_METHOD::LINEAR ? weight - ovr : 0.0f;
      }
      auto& score_j = scores_d[j];
      score_j *= weight;
   }
  }
  auto keep = at::nonzero(scores >= score_thresh).squeeze(1);

  return keep;
}



at::Tensor rotate_nms_cpu(const at::Tensor& dets,
               const at::Tensor& scores,
               const float threshold) {
  at::Tensor result;
  AT_DISPATCH_FLOATING_TYPES(dets.type(), "rotate_nms", [&] {
    result = rotate_nms_cpu_kernel<scalar_t>(dets, scores, threshold);
  });
  return result;
}


std::tuple<at::Tensor, at::Tensor> rotate_soft_nms_cpu(at::Tensor& dets,
               at::Tensor& scores,
               const float nms_thresh,
               const float sigma,
               const float score_thresh,
               const int method
               ) 
{
  auto N = dets.size(0);
  at::Tensor keep;
  at::Tensor indices = at::arange(N, dets.options().dtype(at::kLong).device(at::kCPU));
  if (method == NMS_METHOD::LINEAR || method == NMS_METHOD::GAUSSIAN)
  {
    AT_DISPATCH_FLOATING_TYPES(dets.type(), "rotate_soft_nms", [&] {
      keep = rotate_soft_nms_cpu_kernel<scalar_t>(dets, scores, indices, nms_thresh, sigma, score_thresh, method);
    });
    // auto scores_d = scores.contiguous().data<float>();
    // for (int i = 0; i < scores.size(0); ++i)
    // {
    //   printf("%d) %.3f\n", i, scores_d[i]);
    // }
  } else {
    // original nms
    AT_DISPATCH_FLOATING_TYPES(dets.type(), "nms", [&] {
      keep = rotate_nms_cpu_kernel<scalar_t>(dets, scores, nms_thresh);
    });
  }
  return std::make_tuple(indices, keep);
}