#include <gtest/gtest.h>

#include <unistd.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <tf2/LinearMath/Vector3.h>
#include <tf2_ros/buffer.h>

#include "robot_self_filter/point_hesai.h"
#include "robot_self_filter/point_livox_mid360.h"
#include "robot_self_filter/point_ouster.h"
#include "robot_self_filter/point_pandar.h"
#include "robot_self_filter/point_robosense.h"
#include "robot_self_filter/self_mask.h"
#include "robot_self_filter/self_see_filter.h"

namespace
{

const char kRobotDescription[] = R"(
<robot name="self_filter_test_robot">
  <link name="robot">
    <collision name="sphere">
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><sphere radius="1.0"/></geometry>
    </collision>
    <collision name="box">
      <origin xyz="3 0 0" rpy="0 0 0"/>
      <geometry><box size="1 2 3"/></geometry>
    </collision>
    <collision name="cylinder">
      <origin xyz="-3 0 0" rpy="0 0 0"/>
      <geometry><cylinder radius="0.75" length="2"/></geometry>
    </collision>
  </link>
</robot>)";

std::string uniqueNodeName(const std::string &prefix)
{
  static std::uint64_t sequence = 0;
  return prefix + "_" + std::to_string(++sequence);
}

rclcpp::Node::SharedPtr makeNode(const std::string &prefix)
{
  auto node = std::make_shared<rclcpp::Node>(uniqueNodeName(prefix));
  node->declare_parameter<std::string>("robot_description", kRobotDescription);
  return node;
}

template<typename PointT>
class TestableSelfFilter : public filters::SelfFilter<PointT>
{
public:
  explicit TestableSelfFilter(const rclcpp::Node::SharedPtr &node)
  : filters::SelfFilter<PointT>(node)
  {
  }

  pcl::PointCloud<PointT> render(
    const pcl::PointCloud<PointT> &input,
    const std::vector<int> &mask,
    bool keep_organized,
    bool zero_for_removed_points,
    bool invert)
  {
    this->keep_organized_ = keep_organized;
    this->zero_for_removed_points_ = zero_for_removed_points;
    this->invert_ = invert;
    pcl::PointCloud<PointT> output;
    this->fillResult(input, mask, output);
    return output;
  }
};

template<typename PointT>
PointT makePoint(float x, float y, float z, float intensity)
{
  PointT point{};
  point.x = x;
  point.y = y;
  point.z = z;
  point.intensity = intensity;
  return point;
}

class RclcppEnvironment : public testing::Environment
{
public:
  void SetUp() override
  {
    if (!rclcpp::ok())
      rclcpp::init(0, nullptr);
  }

  void TearDown() override
  {
    if (rclcpp::ok())
      rclcpp::shutdown();
  }
};

testing::Environment * const rclcpp_environment =
  testing::AddGlobalTestEnvironment(new RclcppEnvironment());

TEST(SelfMask, ClassifiesContainmentShadowOutsideMinimumDistanceNaNAndEmptyCloud)
{
  auto node = makeNode("self_mask_classification");
  tf2_ros::Buffer buffer(node->get_clock());
  buffer.setUsingDedicatedThread(true);
  robot_self_filter::LinkInfo link;
  link.name = "robot";
  link.scale = 1.0;
  link.padding = 0.0;
  link.box_scale = {1.0, 1.0, 1.0};
  link.box_padding = {0.0, 0.0, 0.0};
  link.cylinder_scale = {1.0, 1.0};
  link.cylinder_padding = {0.0, 0.0};
  robot_self_filter::SelfMask<pcl::PointXYZ> mask(node, buffer, {link});

  pcl::PointCloud<pcl::PointXYZ> cloud;
  cloud.header.frame_id = "robot";
  cloud.points = {
    pcl::PointXYZ(0.0F, 0.0F, 0.0F),
    pcl::PointXYZ(3.0F, 0.0F, 0.0F),
    pcl::PointXYZ(-3.0F, 0.0F, 0.0F),
    pcl::PointXYZ(5.0F, 0.0F, 0.0F),
    pcl::PointXYZ(0.0F, 5.0F, 0.0F),
    pcl::PointXYZ(0.0F, 0.0F, 2.05F),
    pcl::PointXYZ(std::numeric_limits<float>::quiet_NaN(), 0.0F, 0.0F)};
  cloud.width = static_cast<std::uint32_t>(cloud.points.size());
  cloud.height = 1;

  std::vector<int> result;
  mask.maskIntersection(cloud, tf2::Vector3(0.0, 0.0, 2.0), 0.1, result);
  ASSERT_EQ(result.size(), cloud.points.size());
  EXPECT_EQ(result[0], robot_self_filter::INSIDE);
  EXPECT_EQ(result[1], robot_self_filter::INSIDE);
  EXPECT_EQ(result[2], robot_self_filter::INSIDE);
  EXPECT_EQ(result[3], robot_self_filter::SHADOW);
  EXPECT_EQ(result[4], robot_self_filter::OUTSIDE);
  EXPECT_EQ(result[5], robot_self_filter::INSIDE);
  EXPECT_EQ(result[6], robot_self_filter::OUTSIDE);

  pcl::PointCloud<pcl::PointXYZ> empty;
  empty.header.frame_id = "robot";
  mask.maskIntersection(empty, tf2::Vector3(0.0, 0.0, 2.0), 0.1, result);
  EXPECT_TRUE(result.empty());
}

TEST(SelfMask, ContainmentUsesCollisionOriginsAndPerShapeScale)
{
  auto node = makeNode("self_mask_containment");
  tf2_ros::Buffer buffer(node->get_clock());
  buffer.setUsingDedicatedThread(true);
  robot_self_filter::LinkInfo link;
  link.name = "robot";
  link.scale = 1.5;
  link.padding = 0.1;
  link.box_scale = {2.0, 1.0, 0.5};
  link.box_padding = {0.0, 0.1, 0.2};
  link.cylinder_scale = {1.25, 0.5};
  link.cylinder_padding = {0.05, 0.1};
  robot_self_filter::SelfMask<pcl::PointXYZ> mask(node, buffer, {link});

  pcl::PointCloud<pcl::PointXYZ> cloud;
  cloud.header.frame_id = "robot";
  cloud.points = {
    pcl::PointXYZ(1.59F, 0.0F, 0.0F),
    pcl::PointXYZ(4.0F, 0.0F, 0.0F),
    pcl::PointXYZ(3.0F, 1.09F, 0.0F),
    pcl::PointXYZ(-3.98F, 0.0F, 0.0F),
    pcl::PointXYZ(-3.0F, 0.0F, 0.59F),
    pcl::PointXYZ(0.0F, 4.0F, 0.0F)};
  cloud.width = static_cast<std::uint32_t>(cloud.points.size());
  cloud.height = 1;

  std::vector<int> result;
  mask.maskContainment(cloud, result);
  ASSERT_EQ(result.size(), cloud.points.size());
  EXPECT_EQ(result[0], robot_self_filter::INSIDE);
  EXPECT_EQ(result[1], robot_self_filter::INSIDE);
  EXPECT_EQ(result[2], robot_self_filter::INSIDE);
  EXPECT_EQ(result[3], robot_self_filter::INSIDE);
  EXPECT_EQ(result[4], robot_self_filter::INSIDE);
  EXPECT_EQ(result[5], robot_self_filter::OUTSIDE);
}

TEST(SelfMask, UsesPrefixedNameForTfAndSuffixForUrdfLookup)
{
  auto node = makeNode("self_mask_prefixed_link");
  tf2_ros::Buffer buffer(node->get_clock());
  buffer.setUsingDedicatedThread(true);
  robot_self_filter::LinkInfo link;
  link.name = "agent_7/robot";
  link.scale = 1.0;
  link.padding = 0.0;
  link.box_scale = {1.0, 1.0, 1.0};
  link.box_padding = {0.0, 0.0, 0.0};
  link.cylinder_scale = {1.0, 1.0};
  link.cylinder_padding = {0.0, 0.0};
  robot_self_filter::SelfMask<pcl::PointXYZ> mask(node, buffer, {link});

  std::vector<std::string> link_names;
  mask.getLinkNames(link_names);
  ASSERT_EQ(link_names.size(), 3U);
  EXPECT_EQ(link_names[0], "agent_7/robot");

  pcl::PointCloud<pcl::PointXYZ> cloud;
  cloud.header.frame_id = "agent_7/robot";
  cloud.points = {pcl::PointXYZ(0.0F, 0.0F, 0.0F)};
  cloud.width = 1;
  cloud.height = 1;
  std::vector<int> result;
  mask.maskContainment(cloud, result);
  ASSERT_EQ(result.size(), 1U);
  EXPECT_EQ(result[0], robot_self_filter::INSIDE);
}

// A binary STL of a cube centred on the origin.
std::string writeCubeStl(const std::string &name, const float half)
{
  const float h = half;
  const float corners[8][3] = {
    {-h, -h, -h}, {h, -h, -h}, {h, h, -h}, {-h, h, -h},
    {-h, -h, h}, {h, -h, h}, {h, h, h}, {-h, h, h}};
  const unsigned int faces[12][3] = {
    {0, 2, 1}, {0, 3, 2}, {4, 5, 6}, {4, 6, 7}, {0, 1, 5}, {0, 5, 4},
    {1, 2, 6}, {1, 6, 5}, {2, 3, 7}, {2, 7, 6}, {3, 0, 4}, {3, 4, 7}};
  const std::string path =
    testing::TempDir() + name + "_" + std::to_string(::getpid()) + ".stl";
  std::ofstream out(path, std::ios::binary);
  const char header[80] = {};
  out.write(header, sizeof(header));
  const std::uint32_t count = 12U;
  out.write(reinterpret_cast<const char *>(&count), sizeof(count));
  for (const auto &face : faces)
  {
    const float normal[3] = {0.0F, 0.0F, 0.0F};  // the loader recomputes them
    out.write(reinterpret_cast<const char *>(normal), sizeof(normal));
    for (const unsigned int corner : face)
      out.write(reinterpret_cast<const char *>(corners[corner]), sizeof(corners[corner]));
    const std::uint16_t attribute = 0U;
    out.write(reinterpret_cast<const char *>(&attribute), sizeof(attribute));
  }
  return path;
}

TEST(SelfMask, PadsMeshLinksAndAppliesTheirUrdfScale)
{
  // Modelled in millimetres, as CAD exports often are: a 1 m cube once <mesh scale> applies.
  const std::string stl = writeCubeStl("self_mask_mesh_cube", 500.0F);
  auto node = std::make_shared<rclcpp::Node>(uniqueNodeName("self_mask_mesh"));
  node->declare_parameter<std::string>("robot_description",
    "<robot name=\"mesh_robot\"><link name=\"robot\"><collision><geometry>"
    "<mesh filename=\"file://" + stl + "\" scale=\"0.001 0.001 0.001\"/>"
    "</geometry></collision></link></robot>");
  tf2_ros::Buffer buffer(node->get_clock());
  buffer.setUsingDedicatedThread(true);
  robot_self_filter::LinkInfo link;
  link.name = "robot";
  link.scale = 1.0;
  link.padding = 0.05;
  robot_self_filter::SelfMask<pcl::PointXYZ> mask(node, buffer, {link});
  std::remove(stl.c_str());

  pcl::PointCloud<pcl::PointXYZ> cloud;
  cloud.header.frame_id = "robot";
  cloud.points = {
    pcl::PointXYZ(0.54F, 0.0F, 0.0F),    // inside the padding
    pcl::PointXYZ(0.56F, 0.0F, 0.0F),    // just past it
    pcl::PointXYZ(0.0F, 0.0F, 400.0F)};  // inside only if the scale were dropped
  cloud.width = 3;
  cloud.height = 1;
  std::vector<int> result;
  mask.maskContainment(cloud, result);
  EXPECT_EQ(result, std::vector<int>({
    robot_self_filter::INSIDE, robot_self_filter::OUTSIDE, robot_self_filter::OUTSIDE}));
}

TEST(SelfMask, IgnoresABodyUntilItsTransformArrives)
{
  auto node = std::make_shared<rclcpp::Node>(uniqueNodeName("self_mask_unposed"));
  node->declare_parameter<std::string>("robot_description",
    "<robot name=\"arm_robot\"><link name=\"arm\"><collision><geometry>"
    "<box size=\"1 1 1\"/></geometry></collision></link></robot>");
  tf2_ros::Buffer buffer(node->get_clock());
  buffer.setUsingDedicatedThread(true);
  robot_self_filter::LinkInfo link;
  link.name = "arm";
  link.padding = 0.0;
  robot_self_filter::SelfMask<pcl::PointXYZ> mask(node, buffer, {link});

  pcl::PointCloud<pcl::PointXYZ> cloud;
  cloud.header.frame_id = "sensor";
  cloud.points = {
    pcl::PointXYZ(0.2F, 0.0F, 0.0F),
    pcl::PointXYZ(5.0F, 0.0F, 0.0F),
    pcl::PointXYZ(10.0F, 0.0F, 0.0F),
    pcl::PointXYZ(15.0F, 0.0F, 0.0F)};
  cloud.width = 4;
  cloud.height = 1;

  // No sensor -> arm transform yet. Left at the cloud origin, the box would contain the first
  // point and shadow every other one, since every ray ends at the sensor inside it.
  std::vector<int> result;
  mask.maskIntersection(cloud, tf2::Vector3(0.0, 0.0, 0.0), 0.0, result);
  EXPECT_EQ(result, std::vector<int>(4, robot_self_filter::OUTSIDE));

  geometry_msgs::msg::TransformStamped arm;
  arm.header.frame_id = "sensor";
  arm.child_frame_id = "arm";
  arm.transform.translation.x = 10.0;
  arm.transform.rotation.w = 1.0;
  buffer.setTransform(arm, "test", true);
  mask.maskIntersection(cloud, tf2::Vector3(0.0, 0.0, 0.0), 0.0, result);
  EXPECT_EQ(result, std::vector<int>({
    robot_self_filter::OUTSIDE, robot_self_filter::OUTSIDE,
    robot_self_filter::INSIDE, robot_self_filter::SHADOW}));
}

TEST(SelfFilterOutput, UnorganizedNormalAndInvertedModesPreserveSelectedPoints)
{
  auto node = makeNode("filter_output_unorganized");
  TestableSelfFilter<pcl::PointXYZ> filter(node);
  pcl::PointCloud<pcl::PointXYZ> input;
  input.points = {
    pcl::PointXYZ(1.0F, 2.0F, 3.0F),
    pcl::PointXYZ(4.0F, 5.0F, 6.0F),
    pcl::PointXYZ(7.0F, 8.0F, 9.0F)};
  input.width = 3;
  input.height = 1;
  const std::vector<int> mask = {
    robot_self_filter::OUTSIDE, robot_self_filter::INSIDE, robot_self_filter::SHADOW};

  const auto normal = filter.render(input, mask, false, false, false);
  ASSERT_EQ(normal.size(), 1U);
  EXPECT_FLOAT_EQ(normal[0].x, 1.0F);
  EXPECT_EQ(normal.width, 1U);
  EXPECT_EQ(normal.height, 1U);

  const auto inverted = filter.render(input, mask, false, false, true);
  ASSERT_EQ(inverted.size(), 2U);
  EXPECT_FLOAT_EQ(inverted[0].x, 4.0F);
  EXPECT_FLOAT_EQ(inverted[1].x, 7.0F);
}

TEST(SelfFilterOutput, OrganizedModeUsesConfiguredRemovedPointRepresentation)
{
  auto node = makeNode("filter_output_organized");
  TestableSelfFilter<pcl::PointXYZ> filter(node);
  pcl::PointCloud<pcl::PointXYZ> input;
  input.points = {
    pcl::PointXYZ(1.0F, 2.0F, 3.0F), pcl::PointXYZ(4.0F, 5.0F, 6.0F),
    pcl::PointXYZ(7.0F, 8.0F, 9.0F), pcl::PointXYZ(10.0F, 11.0F, 12.0F)};
  input.width = 2;
  input.height = 2;
  input.header.frame_id = "test_cloud";
  input.is_dense = true;
  input.sensor_origin_ = Eigen::Vector4f(1.0F, 2.0F, 3.0F, 1.0F);
  input.sensor_orientation_ = Eigen::Quaternionf(1.0F, 0.0F, 0.0F, 0.0F);
  const std::vector<int> mask = {
    robot_self_filter::OUTSIDE, robot_self_filter::INSIDE,
    robot_self_filter::SHADOW, robot_self_filter::OUTSIDE};

  const auto nan_output = filter.render(input, mask, true, false, false);
  ASSERT_EQ(nan_output.size(), 4U);
  EXPECT_EQ(nan_output.width, 2U);
  EXPECT_EQ(nan_output.height, 2U);
  EXPECT_EQ(nan_output.header.frame_id, "test_cloud");
  EXPECT_FALSE(nan_output.is_dense);
  EXPECT_TRUE(nan_output.sensor_origin_.isApprox(input.sensor_origin_));
  EXPECT_TRUE(nan_output.sensor_orientation_.isApprox(input.sensor_orientation_));
  EXPECT_FLOAT_EQ(nan_output[0].x, 1.0F);
  EXPECT_TRUE(std::isnan(nan_output[1].x));
  EXPECT_TRUE(std::isnan(nan_output[2].y));
  EXPECT_FLOAT_EQ(nan_output[3].z, 12.0F);

  const auto zero_output = filter.render(input, mask, true, true, false);
  EXPECT_FLOAT_EQ(zero_output[1].x, 0.0F);
  EXPECT_FLOAT_EQ(zero_output[1].y, 0.0F);
  EXPECT_FLOAT_EQ(zero_output[1].z, 0.0F);
  EXPECT_FLOAT_EQ(zero_output[2].x, 0.0F);
  EXPECT_TRUE(zero_output.is_dense);
}

TEST(SelfFilterOutput, RejectsMaskSizeMismatch)
{
  auto node = makeNode("filter_output_mask_mismatch");
  TestableSelfFilter<pcl::PointXYZ> filter(node);
  pcl::PointCloud<pcl::PointXYZ> input;
  input.points = {pcl::PointXYZ(1.0F, 2.0F, 3.0F)};
  input.width = 1;
  input.height = 1;
  EXPECT_THROW(filter.render(input, {}, false, false, false), std::invalid_argument);
}

TEST(SelfFilterOutput, InitializesRemovedCustomPointFieldsDeterministically)
{
  auto node = makeNode("filter_output_custom_blank");
  TestableSelfFilter<PointOuster> filter(node);
  pcl::PointCloud<PointOuster> input;
  PointOuster point = makePoint<PointOuster>(1.0F, 2.0F, 3.0F, 42.0F);
  point.t = 1234U;
  point.reflectivity = 55U;
  point.ring = 7U;
  point.ambient = 88U;
  point.range = 900U;
  input.points = {point};
  input.width = 1;
  input.height = 1;

  const auto output = filter.render(
    input, {robot_self_filter::INSIDE}, true, false, false);
  ASSERT_EQ(output.size(), 1U);
  EXPECT_TRUE(std::isnan(output[0].x));
  EXPECT_FLOAT_EQ(output[0].intensity, 0.0F);
  EXPECT_EQ(output[0].t, 0U);
  EXPECT_EQ(output[0].reflectivity, 0U);
  EXPECT_EQ(output[0].ring, 0U);
  EXPECT_EQ(output[0].ambient, 0U);
  EXPECT_EQ(output[0].range, 0U);
}

template<typename PointT>
void expectCustomPointLayoutIsPreserved()
{
  auto node = makeNode("custom_point_layout");
  TestableSelfFilter<PointT> filter(node);
  pcl::PointCloud<PointT> input;
  input.points = {
    makePoint<PointT>(1.0F, 2.0F, 3.0F, 11.0F),
    makePoint<PointT>(4.0F, 5.0F, 6.0F, 22.0F)};
  input.width = 2;
  input.height = 1;
  const auto output = filter.render(
    input, {robot_self_filter::OUTSIDE, robot_self_filter::INSIDE}, false, false, false);
  ASSERT_EQ(output.size(), 1U);
  EXPECT_FLOAT_EQ(output[0].x, 1.0F);
  EXPECT_FLOAT_EQ(output[0].intensity, 11.0F);
}

TEST(SelfFilterOutput, PreservesOusterPointLayout)
{
  expectCustomPointLayoutIsPreserved<PointOuster>();
}

TEST(SelfFilterOutput, PreservesHesaiPointLayout)
{
  expectCustomPointLayoutIsPreserved<PointHesai>();
}

TEST(SelfFilterOutput, PreservesRobosensePointLayout)
{
  expectCustomPointLayoutIsPreserved<PointRobosense>();
}

TEST(SelfFilterOutput, PreservesPandarPointLayout)
{
  expectCustomPointLayoutIsPreserved<PointPandar>();
}

TEST(SelfFilterOutput, LivoxMid360KeepsEveryFieldOfTheDriversCloud)
{
  // livox_ros_driver2's PointCloud2 (xfer_format 0): packed, so 26 bytes a point and the
  // timestamp unaligned at 18.
  using sensor_msgs::msg::PointField;
  struct Field
  {
    const char *name;
    std::uint32_t offset;
    std::uint8_t datatype;
  };
  const Field fields[] = {
    {"x", 0U, PointField::FLOAT32}, {"y", 4U, PointField::FLOAT32},
    {"z", 8U, PointField::FLOAT32}, {"intensity", 12U, PointField::FLOAT32},
    {"tag", 16U, PointField::UINT8}, {"line", 17U, PointField::UINT8},
    {"timestamp", 18U, PointField::FLOAT64}};
  auto cloud = std::make_shared<sensor_msgs::msg::PointCloud2>();
  cloud->header.frame_id = "robot";
  for (const Field &field : fields)
  {
    PointField point_field;
    point_field.name = field.name;
    point_field.offset = field.offset;
    point_field.datatype = field.datatype;
    point_field.count = 1;
    cloud->fields.push_back(point_field);
  }
  cloud->height = 1;
  cloud->width = 2;
  cloud->point_step = 26;
  cloud->row_step = cloud->point_step * cloud->width;
  cloud->is_dense = true;
  cloud->data.resize(cloud->row_step);
  const auto write = [&cloud](std::size_t point, std::uint32_t offset, const auto value)
  {
    std::memcpy(cloud->data.data() + point * cloud->point_step + offset, &value, sizeof(value));
  };
  for (std::size_t i = 0; i < 2; ++i)
  {
    const float k = static_cast<float>(i);
    write(i, 0U, 1.0F + 3.0F * k);
    write(i, 4U, 2.0F + 3.0F * k);
    write(i, 8U, 3.0F + 3.0F * k);
    write(i, 12U, 40.0F * (k + 1.0F));
    write(i, 16U, static_cast<std::uint8_t>(0x12 + i));
    write(i, 17U, static_cast<std::uint8_t>(3 - i));
    write(i, 18U, 1.5e6 + 1.0e6 * static_cast<double>(i));
  }

  auto node = makeNode("livox_mid360_round_trip");
  filters::SelfFilter<PointLivoxMid360> filter(node);  // no self_see_links: every point stays
  sensor_msgs::msg::PointCloud2 out;
  int input_size = 0;
  int output_size = 0;
  ASSERT_TRUE(filter.fillPointCloud2(cloud, "", out, input_size, output_size));
  ASSERT_EQ(output_size, 2);

  sensor_msgs::PointCloud2ConstIterator<float> x(out, "x");
  sensor_msgs::PointCloud2ConstIterator<float> z(out, "z");
  sensor_msgs::PointCloud2ConstIterator<float> intensity(out, "intensity");
  sensor_msgs::PointCloud2ConstIterator<std::uint8_t> tag(out, "tag");
  sensor_msgs::PointCloud2ConstIterator<std::uint8_t> line(out, "line");
  sensor_msgs::PointCloud2ConstIterator<double> timestamp(out, "timestamp");
  for (std::size_t i = 0; i < 2; ++i, ++x, ++z, ++intensity, ++tag, ++line, ++timestamp)
  {
    const float k = static_cast<float>(i);
    EXPECT_FLOAT_EQ(*x, 1.0F + 3.0F * k);
    EXPECT_FLOAT_EQ(*z, 3.0F + 3.0F * k);
    EXPECT_FLOAT_EQ(*intensity, 40.0F * (k + 1.0F));
    EXPECT_EQ(static_cast<unsigned>(*tag), 0x12U + i);
    EXPECT_EQ(static_cast<unsigned>(*line), 3U - i);
    EXPECT_DOUBLE_EQ(*timestamp, 1.5e6 + 1.0e6 * static_cast<double>(i));
  }
}

}  // namespace
