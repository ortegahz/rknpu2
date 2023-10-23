// Copyright (c) 2021 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "postprocess.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include <set>
#include <vector>
#define LABEL_NALE_TXT_PATH "./model/coco_80_labels_list.txt"
#define NUM_OF_STRDES 4

const int strides[NUM_OF_STRDES] = {8, 16, 32, 64};  // TODO: automatic

static char* labels[OBJ_CLASS_NUM];

// for yolov5
// const int anchor0[6] = {10, 13, 16, 30, 33, 23};
// const int anchor1[6] = {30, 61, 62, 45, 59, 119};
// const int anchor2[6] = {116, 90, 156, 198, 373, 326};

// for yolov7
const int anchor0[6] = {12, 16, 19, 36, 40, 28};
const int anchor1[6] = {36, 75, 76, 55, 72, 146};
const int anchor2[6] = {142, 110, 192, 243, 459, 401};

inline static int clamp(float val, int min, int max) { return val > min ? (val < max ? val : max) : min; }

float __f16_to_f32_s(uint16_t f16)
{
  uint16_t in = f16;

  int32_t t1;
  int32_t t2;
  int32_t t3;
  uint32_t t4;
  float out;

  t1 = in & 0x7fff;         // Non-sign bits
  t2 = in & 0x8000;         // Sign bit
  t3 = in & 0x7c00;         // Exponent

  t1 <<= 13;                // Align mantissa on MSB
  t2 <<= 16;                // Shift sign bit into position

  t1 += 0x38000000;         // Adjust bias

  t1 = (t3 == 0 ? 0 : t1);  // Denormals-as-zero

  t1 |= t2;                 // Re-insert sign bit

  *((uint32_t*)&out) = t1;

  return out;

}

char* readLine(FILE* fp, char* buffer, int* len)
{
  int    ch;
  int    i        = 0;
  size_t buff_len = 0;

  buffer = (char*)malloc(buff_len + 1);
  if (!buffer)
    return NULL; // Out of memory

  while ((ch = fgetc(fp)) != '\n' && ch != EOF) {
    buff_len++;
    void* tmp = realloc(buffer, buff_len + 1);
    if (tmp == NULL) {
      free(buffer);
      return NULL; // Out of memory
    }
    buffer = (char*)tmp;

    buffer[i] = (char)ch;
    i++;
  }
  buffer[i] = '\0';

  *len = buff_len;

  // Detect end
  if (ch == EOF && (i == 0 || ferror(fp))) {
    free(buffer);
    return NULL;
  }
  return buffer;
}

int readLines(const char* fileName, char* lines[], int max_line)
{
  FILE* file = fopen(fileName, "r");
  char* s;
  int   i = 0;
  int   n = 0;

  if (file == NULL) {
    printf("Open %s fail!\n", fileName);
    return -1;
  }

  while ((s = readLine(file, s, &n)) != NULL) {
    lines[i++] = s;
    if (i >= max_line)
      break;
  }
  fclose(file);
  return i;
}

int loadLabelName(const char* locationFilename, char* label[])
{
  printf("loadLabelName %s\n", locationFilename);
  readLines(locationFilename, label, OBJ_CLASS_NUM);
  return 0;
}

static float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1, float ymin1, float xmax1,
                              float ymax1)
{
  float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1) + 1.0);
  float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1) + 1.0);
  float i = w * h;
  float u = (xmax0 - xmin0 + 1.0) * (ymax0 - ymin0 + 1.0) + (xmax1 - xmin1 + 1.0) * (ymax1 - ymin1 + 1.0) - i;
  return u <= 0.f ? 0.f : (i / u);
}

static int nms(int validCount, std::vector<float>& outputLocations, std::vector<int> classIds, std::vector<int>& order,
               int filterId, float threshold)
{
  for (int i = 0; i < validCount; ++i) {
    if (order[i] == -1 || classIds[i] != filterId) {
      continue;
    }
    int n = order[i];
    for (int j = i + 1; j < validCount; ++j) {
      int m = order[j];
      if (m == -1 || classIds[i] != filterId) {
        continue;
      }
      float xmin0 = outputLocations[n * 4 + 0];
      float ymin0 = outputLocations[n * 4 + 1];
      float xmax0 = outputLocations[n * 4 + 0] + outputLocations[n * 4 + 2];
      float ymax0 = outputLocations[n * 4 + 1] + outputLocations[n * 4 + 3];

      float xmin1 = outputLocations[m * 4 + 0];
      float ymin1 = outputLocations[m * 4 + 1];
      float xmax1 = outputLocations[m * 4 + 0] + outputLocations[m * 4 + 2];
      float ymax1 = outputLocations[m * 4 + 1] + outputLocations[m * 4 + 3];

      float iou = CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);

      if (iou > threshold) {
        order[j] = -1;
      }
    }
  }
  return 0;
}

static int quick_sort_indice_inverse(std::vector<float>& input, int left, int right, std::vector<int>& indices)
{
  float key;
  int   key_index;
  int   low  = left;
  int   high = right;
  if (left < right) {
    key_index = indices[left];
    key       = input[left];
    while (low < high) {
      while (low < high && input[high] <= key) {
        high--;
      }
      input[low]   = input[high];
      indices[low] = indices[high];
      while (low < high && input[low] >= key) {
        low++;
      }
      input[high]   = input[low];
      indices[high] = indices[low];
    }
    input[low]   = key;
    indices[low] = key_index;
    quick_sort_indice_inverse(input, left, low - 1, indices);
    quick_sort_indice_inverse(input, low + 1, right, indices);
  }
  return low;
}

static float sigmoid(float x) { return 1.0 / (1.0 + expf(-x)); }

static float unsigmoid(float y) { return -1.0 * logf((1.0 / y) - 1.0); }

inline static int32_t __clip(float val, float min, float max)
{
  float f = val <= min ? min : (val >= max ? max : val);
  return f;
}

static int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale)
{
  float  dst_val = (f32 / scale) + zp;
  int8_t res     = (int8_t)__clip(dst_val, -128, 127);
  return res;
}

static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) { return ((float)qnt - (float)zp) * scale; }

static int process_acfree(int8_t* input_c, int8_t* input_b, int grid_h, int grid_w, int height, int width, int stride,
                   std::vector<float>& boxes, std::vector<float>& objProbs, std::vector<int>& classId, float threshold,
                   int32_t zp_c, float scale_c, int32_t zp_b, float scale_b)
{
  int    validCount = 0;
  int    grid_len   = grid_h * grid_w;
  float  thres      = unsigmoid(threshold);
  int8_t thres_i8_c   = qnt_f32_to_affine(thres, zp_c, scale_c);
  for (int i = 0; i < grid_h; i++) {
    for (int j = 0; j < grid_w; j++) {
          int     offset_c = i * grid_w + j;
          int8_t* in_ptr_c = input_c + offset_c;

          int8_t maxClassProbs = in_ptr_c[0];
          int    maxClassId    = 0;
          for (int k = 1; k < OBJ_CLASS_NUM; ++k) {
            int8_t prob = in_ptr_c[(0 + k) * grid_len];
            if (prob > maxClassProbs) {
              maxClassId    = k;
              maxClassProbs = prob;
            }
          }

          if (maxClassProbs < thres_i8_c) continue;

          int     offset_b = i * grid_w + j;
          int8_t* in_ptr_b = input_b + offset_b;

          float   box_l  = deqnt_affine_to_f32(*in_ptr_b, zp_b, scale_b);
          float   box_t  = deqnt_affine_to_f32(in_ptr_b[grid_len], zp_b, scale_b);
          float   box_r  = deqnt_affine_to_f32(in_ptr_b[2 * grid_len], zp_b, scale_b);
          float   box_b  = deqnt_affine_to_f32(in_ptr_b[3 * grid_len], zp_b, scale_b);
          float box_x1 = j - box_l + 0.5;
          float box_y1 = i - box_t + 0.5;
          float box_x2 = j + box_r + 0.5;
          float box_y2 = i + box_b + 0.5;
          float box_x = box_x1 * (float)stride;
          float box_y = box_y1 * (float)stride;
          float box_w = (box_x2 - box_x1) * (float)stride;
          float box_h = (box_y2 - box_y1) * (float)stride;

          objProbs.push_back(sigmoid(deqnt_affine_to_f32(maxClassProbs, zp_c, scale_c)) * 1.0);
          classId.push_back(maxClassId);
          validCount++;
          boxes.push_back(box_x);
          boxes.push_back(box_y);
          boxes.push_back(box_w);
          boxes.push_back(box_h);
    }
  }
  return validCount;
}

static int process(int8_t* input, int* anchor, int grid_h, int grid_w, int height, int width, int stride,
                   std::vector<float>& boxes, std::vector<float>& objProbs, std::vector<int>& classId, float threshold,
                   int32_t zp, float scale)
{
  int    validCount = 0;
  int    grid_len   = grid_h * grid_w;
  float  thres      = unsigmoid(threshold);
  int8_t thres_i8   = qnt_f32_to_affine(thres, zp, scale);
  for (int a = 0; a < 3; a++) {
    for (int i = 0; i < grid_h; i++) {
      for (int j = 0; j < grid_w; j++) {
        int8_t box_confidence = input[(PROP_BOX_SIZE * a + 4) * grid_len + i * grid_w + j];
        if (box_confidence >= thres_i8) {
          int     offset = (PROP_BOX_SIZE * a) * grid_len + i * grid_w + j;
          int8_t* in_ptr = input + offset;
          float   box_x  = sigmoid(deqnt_affine_to_f32(*in_ptr, zp, scale)) * 2.0 - 0.5;
          float   box_y  = sigmoid(deqnt_affine_to_f32(in_ptr[grid_len], zp, scale)) * 2.0 - 0.5;
          float   box_w  = sigmoid(deqnt_affine_to_f32(in_ptr[2 * grid_len], zp, scale)) * 2.0;
          float   box_h  = sigmoid(deqnt_affine_to_f32(in_ptr[3 * grid_len], zp, scale)) * 2.0;
          box_x          = (box_x + j) * (float)stride;
          box_y          = (box_y + i) * (float)stride;
          box_w          = box_w * box_w * (float)anchor[a * 2];
          box_h          = box_h * box_h * (float)anchor[a * 2 + 1];
          box_x -= (box_w / 2.0);
          box_y -= (box_h / 2.0);

          int8_t maxClassProbs = in_ptr[5 * grid_len];
          int    maxClassId    = 0;
          for (int k = 1; k < OBJ_CLASS_NUM; ++k) {
            int8_t prob = in_ptr[(5 + k) * grid_len];
            if (prob > maxClassProbs) {
              maxClassId    = k;
              maxClassProbs = prob;
            }
          }
          if (maxClassProbs>thres_i8){
            objProbs.push_back(sigmoid(deqnt_affine_to_f32(maxClassProbs, zp, scale))* sigmoid(deqnt_affine_to_f32(box_confidence, zp, scale)));
            classId.push_back(maxClassId);
            validCount++;
            boxes.push_back(box_x);
            boxes.push_back(box_y);
            boxes.push_back(box_w);
            boxes.push_back(box_h);
          }
        }
      }
    }
  }
  return validCount;
}

template<typename t>
int post_process_kps(t *ptInput, std::vector<uint32_t> &qnt_zps, std::vector<float> &qnt_scales, float fCenterX, float fCenterY, float fScaleWT, float fScaleHT, kps_result_group_t *group)
{
  int iGridLen = KPS_OUTPUT_SHAPE_H * KPS_OUTPUT_SHAPE_W;
  for (int i = 0; i < KPS_KEYPOINT_NUM; i++)
  {
    cv::Mat featureMap = cv::Mat::zeros(KPS_OUTPUT_SHAPE_H + 2 * KPS_PIXEL_BORDER, KPS_OUTPUT_SHAPE_W + 2 * KPS_PIXEL_BORDER, CV_32F);
    cv::Mat featureMapFiltered;
    for (int j = 0; j < KPS_OUTPUT_SHAPE_H; j++)
    {
      for (int k = 0; k < KPS_OUTPUT_SHAPE_W; k++)
      {
        float fValue;
        int iOffset = i * iGridLen + j * KPS_OUTPUT_SHAPE_W + k;
        t tValue = ptInput[iOffset];
        printf("std::is_same<t, uint16_t>::value --> %d \n", std::is_same<t, uint16_t>::value);
        printf("std::is_same<t, uint8_t>::value --> %d \n", std::is_same<t, uint8_t>::value);
        if (std::is_same<t, uint16_t>::value) {
          fValue = __f16_to_f32_s(tValue);
        }
        else if (std::is_same<t, uint8_t>::value) {
          fValue = deqnt_affine_to_f32(tValue, qnt_zps[0], qnt_scales[0]);
        }
        else {
          printf("unsupported template type !!! \n");
          exit(1);
        }

        featureMap.at<float>(j + KPS_PIXEL_BORDER, k + KPS_PIXEL_BORDER) = fValue;
      }
    }

    cv::GaussianBlur(featureMap, featureMapFiltered, cv::Size(KPS_GAUSSIAN_KERNEL, KPS_GAUSSIAN_KERNEL), 0);
    char pcPathSave[512];
    sprintf(pcPathSave, "./rknn_output_featureMap_%d.txt", i);
    FILE *pFileHandle = fopen(pcPathSave, "w");
    for (int j = 0; j < featureMapFiltered.rows; j++)
    {
      for (int k = 0; k < featureMapFiltered.cols; k++)
      {
        fprintf(pFileHandle, "%f\n", featureMapFiltered.at<float>(j, k));
      }
    }
    fclose(pFileHandle);

    float fMax = FLT_MIN, fSecondMax = FLT_MIN;
    int iMaxUnpadPosX = -1, iMaxUnpadPosY = -1, iSecondMaxUnpadPosX = -1, iSecondMaxUnpadPosY = -1;
    float fDistance = -1.0, fDeltaX = -1., fDeltaY = -1., fMaxUnpadPosX = -1., fMaxUnpadPosY = -1.;
    for (int j = 0; j < featureMapFiltered.rows; j++)
    {
      for (int k = 0; k < featureMapFiltered.cols; k++)
      {
        float fValue = featureMapFiltered.at<float>(j, k);
        if (fValue > fMax)
        {
          fMax = fValue;
          iMaxUnpadPosX = k - KPS_PIXEL_BORDER;
          iMaxUnpadPosY = j - KPS_PIXEL_BORDER;
        }
      }
    }
    featureMapFiltered.at<float>(iMaxUnpadPosY + KPS_PIXEL_BORDER, iMaxUnpadPosX + KPS_PIXEL_BORDER) = 0.;
    for (int j = 0; j < featureMapFiltered.rows; j++)
    {
      for (int k = 0; k < featureMapFiltered.cols; k++)
      {
        float fValue = featureMapFiltered.at<float>(j, k);
        if (fValue > fSecondMax)
        {
          fSecondMax = fValue;
          iSecondMaxUnpadPosX = k - KPS_PIXEL_BORDER;
          iSecondMaxUnpadPosY = j - KPS_PIXEL_BORDER;
        }
      }
    }
    // printf("featureMapFiltered.at<float>(21, 39) --> %f \n", featureMapFiltered.at<float>(21, 39));
    // printf("[i] iMaxUnpadPosX, iMaxUnpadPosY, iSecondMaxUnpadPosX, fMax iSecondMaxUnpadPosY fSecondMax --> %d, %d, %d, %f, %d, %d %f \n", i, iMaxUnpadPosX, iMaxUnpadPosY, fMax, iSecondMaxUnpadPosX + KPS_PIXEL_BORDER, iSecondMaxUnpadPosY + KPS_PIXEL_BORDER, fSecondMax);
    fDeltaX = (float)(iSecondMaxUnpadPosX - iMaxUnpadPosX);
    fDeltaY = (float)(iSecondMaxUnpadPosY - iMaxUnpadPosY);
    fDistance = sqrt(pow(fDeltaX, 2) + pow(fDeltaY, 2));
    // printf("[i] iMaxUnpadPosX, iMaxUnpadPosY, fDistance --> %d, %d, %d, %f \n", i, iMaxUnpadPosX, iMaxUnpadPosY, fDistance);
    if (fDistance > 1e-3)
    {
      fMaxUnpadPosX = KPS_SHIFTS * fDeltaX / fDistance + (float)iMaxUnpadPosX;
      fMaxUnpadPosY = KPS_SHIFTS * fDeltaY / fDistance + (float)iMaxUnpadPosY;
    }
    // printf("[i] fMaxUnpadPosX, fMaxUnpadPosY, fDistance --> %d, %f, %f, %f \n", i, fMaxUnpadPosX, fMaxUnpadPosY, fDistance);
    fMaxUnpadPosX = std::max((float)0., std::min(fMaxUnpadPosX, (float)(KPS_OUTPUT_SHAPE_W - 1)));
    fMaxUnpadPosY = std::max((float)0., std::min(fMaxUnpadPosY, (float)(KPS_OUTPUT_SHAPE_H - 1)));

    float fConf;
    if (std::is_same<t, uint16_t>::value) {
      float fConf = __f16_to_f32_s(ptInput[i * iGridLen + int(round(fMaxUnpadPosY) + 1e-9) * KPS_OUTPUT_SHAPE_W + int(round(fMaxUnpadPosX) + 1e-9)]) / 255. + 0.5;
    }
    else if (std::is_same<t, uint8_t>::value) {
      fConf = deqnt_affine_to_f32(ptInput[i * iGridLen + int(round(fMaxUnpadPosY) + 1e-9) * KPS_OUTPUT_SHAPE_W + int(round(fMaxUnpadPosX) + 1e-9)], qnt_zps[0], qnt_scales[0]) / 255. + 0.5;
    }
    else {
      printf("unsupported template type !!! \n");
      exit(1);
    }

    fMaxUnpadPosX = fMaxUnpadPosX * KPS_STRIDE + 2;
    fMaxUnpadPosY = fMaxUnpadPosY * KPS_STRIDE + 2;

    // printf("[i] fMaxUnpadPosX, fMaxUnpadPosY, fScaleWT, fScaleHT, fCenterX, fCenterY --> %d, %f, %f, %f, %f, %f, %f \n", i, fMaxUnpadPosX, fMaxUnpadPosY, fScaleWT, fScaleHT, fCenterX, fCenterY);
    fMaxUnpadPosX = fMaxUnpadPosX / (float)KPS_INPUT_SHAPE_W * fScaleWT + fCenterX - fScaleWT * 0.5;
    fMaxUnpadPosY = fMaxUnpadPosY / (float)KPS_INPUT_SHAPE_H * fScaleHT + fCenterY - fScaleHT * 0.5;

    // printf("[i] fMaxUnpadPosX, fMaxUnpadPosY --> %d, %f, %f \n", i, fMaxUnpadPosX, fMaxUnpadPosY);
    // printf("[i] fConf --> %d, %f \n", i, fConf);

    group->results[0].kps[i].x = fMaxUnpadPosX;
    group->results[0].kps[i].y = fMaxUnpadPosY;
    group->results[0].kps[i].conf = fConf;
    group->count = 1;
  }
  return 0;
}

template<typename t>
static int process_player_6(t* input, t* poi, int grid_h, int grid_w, int height, int width, int stride,
                   std::vector<float>& boxes, std::vector<float>& objProbs, std::vector<float>& pois, std::vector<int>& classId, float threshold, uint32_t zp_main, float scale_main, uint32_t zp_aux, float scale_aux)
{
  int    validCount = 0;
  int    grid_len   = grid_h * grid_w;
  float threshold_unsigmoid = unsigmoid(threshold);
  uint8_t thres_i8_c   = qnt_f32_to_affine(threshold_unsigmoid, zp_main, scale_main);
  for (int i = 0; i < grid_h; i++) {
    for (int j = 0; j < grid_w; j++) {
          int     offset_c = 0 * grid_len + i * grid_w + j;
          // uint8_t box_confidence = input[(PROP_BOX_SIZE * a + 4) * grid_len + i * grid_w + j];
          t* in_ptr_c = input + offset_c;
          t maxClassProbs = in_ptr_c[0];
          float maxClassProbs_float;
          // float maxClassProbs_float = __f16_to_f32_s(maxClassProbs);
          if (std::is_same<t, uint16_t>::value) {
            maxClassProbs_float = __f16_to_f32_s(maxClassProbs);
            if (maxClassProbs_float < threshold_unsigmoid) continue;
          }
          else if (std::is_same<t, uint8_t>::value) {
            if (maxClassProbs < thres_i8_c) continue;
            maxClassProbs_float = deqnt_affine_to_f32(maxClassProbs, zp_main, scale_main);
          }
          else {
            printf("unsupported template type !!! \n");
            exit(1);
          }
          int    maxClassId    = 0;

          // if (i == 9 && j == 46)
          // {
          //   printf("[%d %d] maxClassProbs_float --> %f <%d>\n", grid_h, grid_w, maxClassProbs_float, maxClassProbs);
          // }

          // if (maxClassProbs_float < threshold_unsigmoid) continue;

          int     offset_b = 1 * grid_len + i * grid_w + j;
          t* in_ptr_b = input + offset_b;

          // float   box_l  = __f16_to_f32_s(*in_ptr_b);
          // float   box_t  = __f16_to_f32_s(in_ptr_b[grid_len]);
          // float   box_r  = __f16_to_f32_s(in_ptr_b[2 * grid_len]);
          // float   box_b  = __f16_to_f32_s(in_ptr_b[3 * grid_len]);
          float   box_l,box_t,box_r,box_b;
          if (std::is_same<t, uint16_t>::value) {
            box_l  = __f16_to_f32_s(*in_ptr_b);
            box_t  = __f16_to_f32_s(in_ptr_b[grid_len]);
            box_r  = __f16_to_f32_s(in_ptr_b[2 * grid_len]);
            box_b  = __f16_to_f32_s(in_ptr_b[3 * grid_len]);
          }
          else if (std::is_same<t, uint8_t>::value) {
            box_l  = deqnt_affine_to_f32(*in_ptr_b, zp_main, scale_main);
            box_t  = deqnt_affine_to_f32(in_ptr_b[grid_len], zp_main, scale_main);
            box_r  = deqnt_affine_to_f32(in_ptr_b[2 * grid_len], zp_main, scale_main);
            box_b  = deqnt_affine_to_f32(in_ptr_b[3 * grid_len], zp_main, scale_main);
          }
          else {
            printf("unsupported template type !!! \n");
            exit(1);
          }

          float box_x1 = j - box_l + 0.5;
          float box_y1 = i - box_t + 0.5;
          float box_x2 = j + box_r + 0.5;
          float box_y2 = i + box_b + 0.5;
          float box_x = box_x1 * (float)stride;
          float box_y = box_y1 * (float)stride;
          float box_w = (box_x2 - box_x1) * (float)stride;
          float box_h = (box_y2 - box_y1) * (float)stride;

          int pp_x = -1;
          int pp_y = -1;
          float pp_c = unsigmoid(0.0);
          int idx_base = 0;
          for (int p0 = 0; p0 < NUM_OF_STRDES; p0++)
          {
            int stride_pick = strides[p0];
            int sp = width / stride_pick;
            int dp = box_h / stride_pick;
            int ep = box_w / stride_pick;
            int px = box_x / stride_pick;
            int py = box_y / stride_pick;
            // printf("stride_pick --> %d\n", stride_pick);
            // printf("sp --> %d\n", sp);
            // printf("idx_base --> %d\n", idx_base);
            for (int p1 = 0; p1 <= dp; p1++)
            {
              for (int p2 = 0; p2 <= ep; p2++)
              {
                int sx = p2 + px; int sy =  py + p1;
                int  offset_p = idx_base + sy * sp + sx;
                t* in_ptr_p = poi + offset_p;
                // float value = __f16_to_f32_s(in_ptr_p[0]);
                float value;
                if (std::is_same<t, uint16_t>::value) {
                  value = __f16_to_f32_s(in_ptr_p[0]);
                }
                else if (std::is_same<t, uint8_t>::value) {
                  value = qnt_f32_to_affine(in_ptr_p[0], zp_aux, scale_aux);
                }
                else {
                  printf("unsupported template type !!! \n");
                  exit(1);
                }
                // if (i == 46 && j == 54)
                // {
                //   printf("sx --> %d and sy --> %d\n", sx, sy);
                // }
                // if ((sx == 111) && (sy == 97) && (p0 == 0))
                // {
                //   printf("value_sigmoid_pre --> %f\n", sigmoid(value) * 1.0);
                // }
                if (value > pp_c)
                {
                  pp_x = (p2 + px) * stride_pick;
                  pp_y = (py + p1) * stride_pick;
                  pp_c = value;
                }
              }
            }
            idx_base += width / stride_pick * height / stride_pick;
          }

          // if (i == 46 && j == 54)
          // {
          //   int  offset_p = 0 + 97 * (width / 8) + 111;
          //   uint16_t* in_ptr_p = poi + offset_p;
          //   float value = __f16_to_f32_s(in_ptr_p[0]);
          //   printf("value_sigmoid --> %f\n", sigmoid(value) * 1.0);
          //   printf("[%d %d] maxClassProbs_float_sigmoid --> %f <%d>\n", grid_h, grid_w, sigmoid(maxClassProbs_float), maxClassProbs);
          //   printf("pp_x pp_y pp_c --> %d, %d, %f\n", pp_x, pp_y, sigmoid(pp_c) * 1.0);
          // }

          objProbs.push_back(sigmoid(maxClassProbs_float) * 1.0);
          classId.push_back(maxClassId);
          validCount++;
          boxes.push_back(box_x);
          boxes.push_back(box_y);
          boxes.push_back(box_w);
          boxes.push_back(box_h);
          pois.push_back(float(pp_x));
          pois.push_back(float(pp_y));
          pois.push_back(sigmoid(pp_c) * 1.0);
    }
  }
  return validCount;
}

template<typename t>
int post_process_player_6(t* input0, t* input1, t* input2, t* input3, t* input4, int model_in_h, int model_in_w, float conf_threshold, float nms_threshold, float scale_w, float scale_h, std::vector<int32_t>& qnt_zps, std::vector<float>& qnt_scales, detect_result_group_float_t* group)
{
  printf("std::is_same<t, uint16_t>::value --> %d \n", std::is_same<t, uint16_t>::value);
  printf("std::is_same<t, uint8_t>::value --> %d \n", std::is_same<t, uint8_t>::value);
  static int init = -1;
  if (init == -1) {
    int ret = 0;
    ret     = loadLabelName(LABEL_NALE_TXT_PATH, labels);
    if (ret < 0) {
      return -1;
    }

    init = 0;
  }
  memset(group, 0, sizeof(detect_result_group_float_t));

  std::vector<float> filterBoxes;
  std::vector<float> objProbs;
  std::vector<int>   classId;
  std::vector<float> pois;

  printf("conf_threshold --> %f \n", conf_threshold);

  // stride 8
  int stride0     = 8;
  int grid_h0     = model_in_h / stride0;
  int grid_w0     = model_in_w / stride0;
  int validCount0 = 0;
  validCount0 = process_player_6(input0, input4, grid_h0, grid_w0, model_in_h, model_in_w, stride0, filterBoxes, objProbs, pois,
                        classId, conf_threshold, qnt_zps[0], qnt_scales[0], qnt_zps[4], qnt_scales[4]);

  printf("validCount0 --> %d \n", validCount0);

  // stride 16
  int stride1     = 16;
  int grid_h1     = model_in_h / stride1;
  int grid_w1     = model_in_w / stride1;
  int validCount1 = 0;
  validCount1 = process_player_6(input1, input4, grid_h1, grid_w1, model_in_h, model_in_w, stride1, filterBoxes, objProbs, pois,
                        classId, conf_threshold, qnt_zps[1], qnt_scales[1], qnt_zps[4], qnt_scales[4]);
  
  printf("validCount1 --> %d \n", validCount1);

  // stride 32
  int stride2     = 32;
  int grid_h2     = model_in_h / stride2;
  int grid_w2     = model_in_w / stride2;
  int validCount2 = 0;
  validCount2 = process_player_6(input2, input4, grid_h2, grid_w2, model_in_h, model_in_w, stride2, filterBoxes, objProbs, pois,
                        classId, conf_threshold, qnt_zps[2], qnt_scales[2], qnt_zps[4], qnt_scales[4]);

  printf("validCount2 --> %d \n", validCount2);

  // stride 64
  int stride3     = 64;
  int grid_h3     = model_in_h / stride3;
  int grid_w3     = model_in_w / stride3;
  int validCount3 = 0;
  validCount3 = process_player_6(input3, input4, grid_h3, grid_w3, model_in_h, model_in_w, stride3, filterBoxes, objProbs, pois,
                        classId, conf_threshold, qnt_zps[3], qnt_scales[3], qnt_zps[4], qnt_scales[4]);

  printf("validCount3 --> %d \n", validCount3);

  int validCount = validCount0 + validCount1 + validCount2 + validCount3;

  printf("validCount --> %d \n", validCount);
  // no object detect
  if (validCount <= 0) {
    return 0;
  }

  std::vector<int> indexArray;
  for (int i = 0; i < validCount; ++i) {
    indexArray.push_back(i);
  }

  quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray);

  std::set<int> class_set(std::begin(classId), std::end(classId));

  for (auto c : class_set) {
    nms(validCount, filterBoxes, classId, indexArray, c, nms_threshold);
  }

  int last_count = 0;
  group->count   = 0;
  /* box valid detect target */
  for (int i = 0; i < validCount; ++i) {
    if (indexArray[i] == -1 || last_count >= OBJ_NUMB_MAX_SIZE) {
      continue;
    }
    int n = indexArray[i];

    float x1       = filterBoxes[n * 4 + 0];
    float y1       = filterBoxes[n * 4 + 1];
    float x2       = x1 + filterBoxes[n * 4 + 2];
    float y2       = y1 + filterBoxes[n * 4 + 3];
    int   id       = classId[n];
    float obj_conf = objProbs[i];
    float poi_x       = pois[n * 3 + 0];
    float poi_y       = pois[n * 3 + 1];
    float poi_c       = pois[n * 3 + 2];

    // printf("pp_x pp_y pp_c --> %f, %f, %f\n", poi_x, poi_y, poi_c);

    group->results[last_count].box.left   = clamp(x1, 0, model_in_w) / scale_w;
    group->results[last_count].box.top    = clamp(y1, 0, model_in_h) / scale_h;
    group->results[last_count].box.right  = clamp(x2, 0, model_in_w) / scale_w;
    group->results[last_count].box.bottom = clamp(y2, 0, model_in_h) / scale_h;
    group->results[last_count].prop       = obj_conf;
    group->results[last_count].poi.x = poi_x / scale_w;
    group->results[last_count].poi.y = poi_y / scale_h;
    group->results[last_count].poi.conf = poi_c;
    char* label                           = labels[id];
    strncpy(group->results[last_count].name, label, OBJ_NAME_MAX_SIZE);

    // printf("result %2d: (%4d, %4d, %4d, %4d), %s\n", i, group->results[last_count].box.left,
    // group->results[last_count].box.top,
    //        group->results[last_count].box.right, group->results[last_count].box.bottom, label);
    last_count++;
  }
  group->count = last_count;

  printf("last_count --> %d \n", last_count);

  return 0;
}

template int post_process_player_6 <uint16_t > (uint16_t* input0, uint16_t* input1, uint16_t* input2, uint16_t* input3, uint16_t* input4, int model_in_h, int model_in_w, float conf_threshold, float nms_threshold, float scale_w, float scale_h, std::vector<int32_t>& qnt_zps, std::vector<float>& qnt_scales, detect_result_group_float_t* group);

template int post_process_player_6 <int8_t > (int8_t* input0, int8_t* input1, int8_t* input2, int8_t* input3, int8_t* input4, int model_in_h, int model_in_w, float conf_threshold, float nms_threshold, float scale_w, float scale_h, std::vector<int32_t>& qnt_zps, std::vector<float>& qnt_scales, detect_result_group_float_t* group);

int post_process_kps_wrapper(rknn_context ctx_kps, cv::Mat *Img, pcBOX_RECT_FLOAT stBoxRect, void *resize_buf, rknn_tensor_attr *output_attrs, kps_result_group_t *pKps_result_group, bool bF16)
{
  // Load image
  // CImg<unsigned char> img_kps("./model/rsn_align.bmp");
  unsigned char *input_data = NULL;
  // input_data = load_image("./model/rsn_align.bmp", &img_height_kps, &img_width_kps, &img_channel_kps, &input_attrs[0]);
  // if (!input_data)
  // {
  //   return -1;
  // }

  float fCenterX = (stBoxRect.left + stBoxRect.right) / 2.;
  float fCenterY = (stBoxRect.top + stBoxRect.bottom) / 2.;
  float fScaleW = (stBoxRect.right - stBoxRect.left) * 1.0 / KPS_PIXEL_STD;
  float fScaleH = (stBoxRect.bottom - stBoxRect.top) * 1.0 / KPS_PIXEL_STD;
  fScaleW *= (1. + KPS_X_EXTENTION);
  fScaleH *= (1. + KPS_Y_EXTENTION);
  // printf("fScaleW, fScaleH --> %f, %f \n", fScaleW, fScaleH);
  // printf("KPS_INPUT_SHAPE_W / KPS_INPUT_SHAPE_H  * fScaleH -- > %f \n", (KPS_INPUT_SHAPE_W / KPS_INPUT_SHAPE_H  * fScaleH));
  if (fScaleW > KPS_WIDTH_HEIGHT_RATIO * fScaleH)
  {
    printf("true \n");
    fScaleH = fScaleW * 1.0 / KPS_WIDTH_HEIGHT_RATIO;
  }
  else
  {
    printf("false \n");
    fScaleW = fScaleH * 1.0 * KPS_WIDTH_HEIGHT_RATIO;
  }
  printf("fScaleW, fScaleH --> %f, %f \n", fScaleW, fScaleH);
  float fScaleWT = fScaleW * KPS_PIXEL_STD;
  float fScaleHT = fScaleH * KPS_PIXEL_STD;
  float fSrcW = fScaleWT;
  float fDstW = KPS_INPUT_SHAPE_W;
  float fDstH = KPS_INPUT_SHAPE_H;

  cv::Point2f srcTri[3];
  cv::Point2f dstTri[3];
  srcTri[0] = cv::Point2f(fCenterX, fCenterY);
  srcTri[1] = cv::Point2f(fCenterX, fCenterY - fSrcW * 0.5);
  srcTri[2] = cv::Point2f(fCenterX - fSrcW * 0.5, fCenterY - fSrcW * 0.5);
  dstTri[0] = cv::Point2f(fDstW * 0.5, fDstH * 0.5);
  dstTri[1] = cv::Point2f(fDstW * 0.5, fDstH * 0.5 - fDstW * 0.5);
  dstTri[2] = cv::Point2f(fDstW * 0.5 - fDstW * 0.5, fDstH * 0.5 - fDstW * 0.5);
  printf("srcTri[0] --> %f, %f \n", fCenterX, fCenterY);
  printf("srcTri[1] --> %f, %f \n", fCenterX, fCenterY - fSrcW * 0.5);
  printf("srcTri[2] --> %f, %f \n", fCenterX - fSrcW * 0.5, fCenterY - fSrcW * 0.5);
  printf("dstTri[0] --> %f, %f \n", fDstW * 0.5, fDstH * 0.5);
  printf("dstTri[1] --> %f, %f \n", fDstW * 0.5, fDstH * 0.5 - fDstW * 0.5);
  printf("dstTri[2] --> %f, %f \n", fDstW * 0.5 - fDstW * 0.5, fDstH * 0.5 - fDstW * 0.5);
  cv::Mat Trans(2, 3, CV_32FC1);
  Trans = cv::getAffineTransform(srcTri, dstTri);

  // cv::Mat img = cv::imread("./model/rsn_align.bmp");
  // cv::Mat Img = cv::imread("./model/rsn.bmp");
  cv::Mat ImgWA;
  cv::warpAffine(*Img, ImgWA, Trans, cv::Size(KPS_INPUT_SHAPE_W, KPS_INPUT_SHAPE_H));
  cv::imwrite("./ImgWA.bmp", ImgWA);
  input_data = ImgWA.data;

  // save data
  char acSavePath[512];
  sprintf(acSavePath, "./rknn_output_input_data.txt");
  FILE *pFileHandle = fopen(acSavePath, "w");
  for (int i = 0; i < KPS_INPUT_SHAPE_H; i++)
  {
    for (int j = 0; j < KPS_INPUT_SHAPE_W; j++)
    {
      for (int k = 0; k < 3; k++)
      {
        fprintf(pFileHandle, "%u\n", input_data[i * KPS_INPUT_SHAPE_W * 3 + j * 3 + k]);
      }
    }
  }
  fclose(pFileHandle);

  rknn_input inputs[1];
  memset(inputs, 0, sizeof(inputs));
  inputs[0].index = 0;
  inputs[0].type = RKNN_TENSOR_UINT8;
  inputs[0].size = KPS_INPUT_SHAPE_H * KPS_INPUT_SHAPE_W * 3;
  inputs[0].fmt = RKNN_TENSOR_NHWC;
  inputs[0].pass_through = 0;

  memcpy(resize_buf, input_data, KPS_INPUT_SHAPE_H * KPS_INPUT_SHAPE_W * 3);

  inputs[0].buf = resize_buf;
  rknn_inputs_set(ctx_kps, 1, inputs);

  rknn_output outputs[1];
  memset(outputs, 0, sizeof(outputs));
  for (int i = 0; i < 1; i++)
  {
    outputs[i].want_float = 0;
  }

  int ret = rknn_run(ctx_kps, NULL);
  ret = rknn_outputs_get(ctx_kps, 1, outputs, NULL);

  std::vector<float> out_scales;
  std::vector<uint32_t> out_zps;
  for (int i = 0; i < 1; ++i)
  {
    out_scales.push_back(output_attrs[i].scale);
    out_zps.push_back(output_attrs[i].zp);
  }

  // save float outputs for debugging
  if (bF16) {
    for (int i = 0; i < 1; ++i)
    {
      char path[512];
      sprintf(path, "./rknn_output_real_kps_nq_%d.txt", i);
      FILE *fp = fopen(path, "w");
      uint16_t *output = (uint16_t *)outputs[i].buf;
      uint32_t n_elems = output_attrs[i].n_elems;
      for (int j = 0; j < n_elems; j++)
      {
        float value = __f16_to_f32_s(output[j]);
        fprintf(fp, "%f\n", value);
      }
      fclose(fp);
    }
  }
  else {
    for (int i = 0; i < 1; ++i)
    {
      char path[512];
      sprintf(path, "./rknn_output_real_kps_%d.txt", i);
      FILE *fp = fopen(path, "w");
      uint8_t *output = (uint8_t *)outputs[i].buf;
      float out_scale = output_attrs[i].scale;
      uint32_t out_zp = output_attrs[i].zp;
      uint32_t n_elems = output_attrs[i].n_elems;
      // printf("output idx %d n_elems --> %d \n", i, n_elems);
      for (int j = 0; j < n_elems; j++)
      {
        float value = deqnt_affine_to_f32(output[j], out_zp, out_scale);
        fprintf(fp, "%f\n", value);
      }
      fclose(fp);
    }
  }

  // post_process_acfree((uint8_t*)outputs[0].buf, (uint8_t*)outputs[1].buf, (uint8_t*)outputs[2].buf, (uint8_t*)outputs[3].buf, (uint8_t*)outputs[4].buf, (uint8_t*)outputs[5].buf, height, width, box_conf_threshold, nms_threshold, scale_w, scale_h, out_zps, out_scales, &detect_result_group);
  // post_process_kps_f16((uint8_t*)outputs[0].buf, fCenterX, fCenterY, fScaleWT, fScaleHT, pKps_result_group);
  if (bF16) {
    post_process_kps((uint16_t *)outputs[0].buf, out_zps, out_scales, fCenterX, fCenterY, fScaleWT, fScaleHT, pKps_result_group);
  }
  else {
    post_process_kps((uint8_t *)outputs[0].buf, out_zps, out_scales, fCenterX, fCenterY, fScaleWT, fScaleHT, pKps_result_group);
  }

  // Save KPS Parser Results
  FILE *fid = fopen("npu_parser_results_kps.txt", "w");
  assert(fid != NULL);
  for (int i = 0; i < pKps_result_group->count; i++)
  {
    kps_result_t *kps_result = &(pKps_result_group->results[i]);
    for (int j = 0; j < KPS_KEYPOINT_NUM; j++)
    {
      float x = (float)kps_result->kps[j].x;
      float y = (float)kps_result->kps[j].y;
      float conf = kps_result->kps[j].conf;
      fprintf(fid, "%f, %f,  %f \n", x, y, conf);
      // fprintf(fid, "%f, %f \n", x, y);
    }
  }
  fclose(fid);

  return 0;
}

int post_process_acfree(int8_t* input0, int8_t* input1, int8_t* input2, int8_t* input3, int8_t* input4, int8_t* input5, int model_in_h, int model_in_w, float conf_threshold,
                 float nms_threshold, float scale_w, float scale_h, std::vector<int32_t>& qnt_zps,
                 std::vector<float>& qnt_scales, detect_result_group_t* group)
{
  static int init = -1;
  if (init == -1) {
    int ret = 0;
    ret     = loadLabelName(LABEL_NALE_TXT_PATH, labels);
    if (ret < 0) {
      return -1;
    }

    init = 0;
  }
  memset(group, 0, sizeof(detect_result_group_t));

  std::vector<float> filterBoxes;
  std::vector<float> objProbs;
  std::vector<int>   classId;

  // stride 8
  int stride0     = 8;
  int grid_h0     = model_in_h / stride0;
  int grid_w0     = model_in_w / stride0;
  int validCount0 = 0;
  validCount0 = process_acfree(input0, input3, grid_h0, grid_w0, model_in_h, model_in_w, stride0, filterBoxes, objProbs,
                        classId, conf_threshold, qnt_zps[0], qnt_scales[0], qnt_zps[3], qnt_scales[3]);

  // stride 16
  int stride1     = 16;
  int grid_h1     = model_in_h / stride1;
  int grid_w1     = model_in_w / stride1;
  int validCount1 = 0;
  validCount1 = process_acfree(input1, input4, grid_h1, grid_w1, model_in_h, model_in_w, stride1, filterBoxes, objProbs,
                        classId, conf_threshold, qnt_zps[1], qnt_scales[1], qnt_zps[4], qnt_scales[4]);

  // stride 32
  int stride2     = 32;
  int grid_h2     = model_in_h / stride2;
  int grid_w2     = model_in_w / stride2;
  int validCount2 = 0;
  validCount2 = process_acfree(input2, input5, grid_h2, grid_w2, model_in_h, model_in_w, stride2, filterBoxes, objProbs,
                        classId, conf_threshold, qnt_zps[2], qnt_scales[2], qnt_zps[5], qnt_scales[5]);

  int validCount = validCount0 + validCount1 + validCount2;
  // no object detect
  if (validCount <= 0) {
    return 0;
  }

  std::vector<int> indexArray;
  for (int i = 0; i < validCount; ++i) {
    indexArray.push_back(i);
  }

  quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray);

  std::set<int> class_set(std::begin(classId), std::end(classId));

  for (auto c : class_set) {
    nms(validCount, filterBoxes, classId, indexArray, c, nms_threshold);
  }

  int last_count = 0;
  group->count   = 0;
  /* box valid detect target */
  for (int i = 0; i < validCount; ++i) {
    if (indexArray[i] == -1 || last_count >= OBJ_NUMB_MAX_SIZE) {
      continue;
    }
    int n = indexArray[i];

    float x1       = filterBoxes[n * 4 + 0];
    float y1       = filterBoxes[n * 4 + 1];
    float x2       = x1 + filterBoxes[n * 4 + 2];
    float y2       = y1 + filterBoxes[n * 4 + 3];
    int   id       = classId[n];
    float obj_conf = objProbs[i];

    group->results[last_count].box.left   = (int)(clamp(x1, 0, model_in_w) / scale_w);
    group->results[last_count].box.top    = (int)(clamp(y1, 0, model_in_h) / scale_h);
    group->results[last_count].box.right  = (int)(clamp(x2, 0, model_in_w) / scale_w);
    group->results[last_count].box.bottom = (int)(clamp(y2, 0, model_in_h) / scale_h);
    group->results[last_count].prop       = obj_conf;
    char* label                           = labels[id];
    strncpy(group->results[last_count].name, label, OBJ_NAME_MAX_SIZE);

    // printf("result %2d: (%4d, %4d, %4d, %4d), %s\n", i, group->results[last_count].box.left,
    // group->results[last_count].box.top,
    //        group->results[last_count].box.right, group->results[last_count].box.bottom, label);
    last_count++;
  }
  group->count = last_count;

  return 0;
}

int post_process(int8_t* input0, int8_t* input1, int8_t* input2, int model_in_h, int model_in_w, float conf_threshold,
                 float nms_threshold, float scale_w, float scale_h, std::vector<int32_t>& qnt_zps,
                 std::vector<float>& qnt_scales, detect_result_group_t* group)
{
  static int init = -1;
  if (init == -1) {
    int ret = 0;
    ret     = loadLabelName(LABEL_NALE_TXT_PATH, labels);
    if (ret < 0) {
      return -1;
    }

    init = 0;
  }
  memset(group, 0, sizeof(detect_result_group_t));

  std::vector<float> filterBoxes;
  std::vector<float> objProbs;
  std::vector<int>   classId;

  // stride 8
  int stride0     = 8;
  int grid_h0     = model_in_h / stride0;
  int grid_w0     = model_in_w / stride0;
  int validCount0 = 0;
  validCount0 = process(input0, (int*)anchor0, grid_h0, grid_w0, model_in_h, model_in_w, stride0, filterBoxes, objProbs,
                        classId, conf_threshold, qnt_zps[0], qnt_scales[0]);

  // stride 16
  int stride1     = 16;
  int grid_h1     = model_in_h / stride1;
  int grid_w1     = model_in_w / stride1;
  int validCount1 = 0;
  validCount1 = process(input1, (int*)anchor1, grid_h1, grid_w1, model_in_h, model_in_w, stride1, filterBoxes, objProbs,
                        classId, conf_threshold, qnt_zps[1], qnt_scales[1]);

  // stride 32
  int stride2     = 32;
  int grid_h2     = model_in_h / stride2;
  int grid_w2     = model_in_w / stride2;
  int validCount2 = 0;
  validCount2 = process(input2, (int*)anchor2, grid_h2, grid_w2, model_in_h, model_in_w, stride2, filterBoxes, objProbs,
                        classId, conf_threshold, qnt_zps[2], qnt_scales[2]);

  int validCount = validCount0 + validCount1 + validCount2;
  // no object detect
  if (validCount <= 0) {
    return 0;
  }

  std::vector<int> indexArray;
  for (int i = 0; i < validCount; ++i) {
    indexArray.push_back(i);
  }

  quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray);

  std::set<int> class_set(std::begin(classId), std::end(classId));

  for (auto c : class_set) {
    nms(validCount, filterBoxes, classId, indexArray, c, nms_threshold);
  }

  int last_count = 0;
  group->count   = 0;
  /* box valid detect target */
  for (int i = 0; i < validCount; ++i) {
    if (indexArray[i] == -1 || last_count >= OBJ_NUMB_MAX_SIZE) {
      continue;
    }
    int n = indexArray[i];

    float x1       = filterBoxes[n * 4 + 0];
    float y1       = filterBoxes[n * 4 + 1];
    float x2       = x1 + filterBoxes[n * 4 + 2];
    float y2       = y1 + filterBoxes[n * 4 + 3];
    int   id       = classId[n];
    float obj_conf = objProbs[i];

    group->results[last_count].box.left   = (int)(clamp(x1, 0, model_in_w) / scale_w);
    group->results[last_count].box.top    = (int)(clamp(y1, 0, model_in_h) / scale_h);
    group->results[last_count].box.right  = (int)(clamp(x2, 0, model_in_w) / scale_w);
    group->results[last_count].box.bottom = (int)(clamp(y2, 0, model_in_h) / scale_h);
    group->results[last_count].prop       = obj_conf;
    char* label                           = labels[id];
    strncpy(group->results[last_count].name, label, OBJ_NAME_MAX_SIZE);

    // printf("result %2d: (%4d, %4d, %4d, %4d), %s\n", i, group->results[last_count].box.left,
    // group->results[last_count].box.top,
    //        group->results[last_count].box.right, group->results[last_count].box.bottom, label);
    last_count++;
  }
  group->count = last_count;

  return 0;
}

void deinitPostProcess()
{
  for (int i = 0; i < OBJ_CLASS_NUM; i++) {
    if (labels[i] != nullptr) {
      free(labels[i]);
      labels[i] = nullptr;
    }
  }
}
