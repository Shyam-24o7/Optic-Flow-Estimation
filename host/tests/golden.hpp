// Reader for golden vectors written by tools/export_golden.py (OpenCV FileStorage).
#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace golden {

inline std::string path(const std::string& name) { return std::string(FCW_GOLDEN_DIR) + "/" + name + ".yml.gz"; }

class File {
 public:
  explicit File(const std::string& name) : fs_(path(name), cv::FileStorage::READ) {
    if (!fs_.isOpened()) throw std::runtime_error("golden file missing: " + path(name) + " (run tools/export_golden.py)");
  }
  cv::FileNode node(const std::string& key) const {
    cv::FileNode n = fs_[key];
    if (n.empty()) throw std::runtime_error("golden key missing: " + key);
    return n;
  }
  cv::Mat mat(const std::string& key) const { cv::Mat m; node(key) >> m; return m; }
  double num(const std::string& key) const { return static_cast<double>(node(key)); }
  // int64 values are stored as decimal strings: FileStorage ints are 32-bit.
  int64_t i64(const std::string& key) const { return std::stoll(static_cast<std::string>(node(key))); }
  std::string str(const std::string& key) const { return static_cast<std::string>(node(key)); }
  // Images are PNG files next to the YAML; the key holds the file name.
  cv::Mat image(const std::string& key) const {
    cv::Mat img = cv::imread(std::string(FCW_GOLDEN_DIR) + "/" + str(key), cv::IMREAD_UNCHANGED);
    if (img.empty()) throw std::runtime_error("golden image missing: " + str(key));
    return img;
  }
  bool has(const std::string& key) const { return !fs_[key].empty(); }

 private:
  cv::FileStorage fs_;
};

}  // namespace golden
