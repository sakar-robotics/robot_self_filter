#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Vector3.h>

#include "robot_self_filter/bodies.h"
#include "robot_self_filter/shapes.h"

namespace
{

constexpr double kTolerance = 1e-9;

bool legacyBoxIntersectsRay(
  const robot_self_filter::bodies::Box &body,
  const tf2::Vector3 &origin,
  const tf2::Vector3 &direction,
  std::vector<tf2::Vector3> *intersections)
{
  const tf2::Vector3 center = body.getPose().getOrigin();
  const tf2::Matrix3x3 basis(body.getPose().getBasis());
  const tf2::Vector3 normals[3] = {
    basis.getColumn(0), basis.getColumn(1), basis.getColumn(2)};
  const double half_extents[3] = {
    body.getScaledHalfLength(), body.getScaledHalfWidth(), body.getScaledHalfHeight()};
  const tf2::Vector3 corner_1 = center - normals[0] * half_extents[0] -
    normals[1] * half_extents[1] - normals[2] * half_extents[2];
  const tf2::Vector3 corner_2 = center + normals[0] * half_extents[0] +
    normals[1] * half_extents[1] + normals[2] * half_extents[2];

  double t_near = -std::numeric_limits<double>::infinity();
  double t_far = std::numeric_limits<double>::infinity();
  for (int i = 0; i < 3; ++i)
  {
    const double direction_projection = normals[i].dot(direction);
    const double minimum = corner_1.dot(normals[i]);
    const double maximum = corner_2.dot(normals[i]);
    if (std::fabs(direction_projection) > 1e-9)
    {
      double t1 = (minimum - normals[i].dot(origin)) / direction_projection;
      double t2 = (maximum - normals[i].dot(origin)) / direction_projection;
      if (t1 > t2)
        std::swap(t1, t2);
      t_near = std::max(t_near, t1);
      t_far = std::min(t_far, t2);
      if (t_near > t_far || t_far < 0.0)
        return false;
    }
    else
    {
      const double value = normals[i].dot(origin);
      if (value < minimum || value > maximum)
        return false;
    }
  }

  if (intersections)
  {
    if ((t_far - t_near) > 1e-9)
      intersections->push_back(origin + direction * t_near);
    else
      intersections->push_back(origin + direction * t_far);
  }
  return true;
}

std::uint32_t nextRandom(std::uint32_t &state)
{
  state = state * 1664525U + 1013904223U;
  return state;
}

double randomRange(std::uint32_t &state, double minimum, double maximum)
{
  const double unit = static_cast<double>(nextRandom(state)) /
    static_cast<double>(std::numeric_limits<std::uint32_t>::max());
  return minimum + unit * (maximum - minimum);
}

TEST(Bodies, SphereContainmentIntersectionAndBoundingSphere)
{
  robot_self_filter::shapes::Sphere shape(1.0);
  robot_self_filter::bodies::Sphere body(&shape);
  body.setScale(2.0);
  body.setPadding(0.25);

  tf2::Transform pose;
  pose.setIdentity();
  pose.setOrigin(tf2::Vector3(1.0, 2.0, 3.0));
  body.setPose(pose);

  EXPECT_TRUE(body.containsPoint(tf2::Vector3(3.0, 2.0, 3.0)));
  EXPECT_FALSE(body.containsPoint(tf2::Vector3(3.25, 2.0, 3.0)));
  EXPECT_FALSE(body.containsPoint(tf2::Vector3(3.250001, 2.0, 3.0)));

  std::vector<tf2::Vector3> hits;
  ASSERT_TRUE(body.intersectsRay(
    tf2::Vector3(-5.0, 2.0, 3.0), tf2::Vector3(1.0, 0.0, 0.0), &hits, 1));
  ASSERT_EQ(hits.size(), 1U);
  EXPECT_NEAR(hits.front().x(), -1.25, kTolerance);

  robot_self_filter::bodies::BoundingSphere bound;
  body.computeBoundingSphere(bound);
  EXPECT_NEAR(bound.center.x(), 1.0, kTolerance);
  EXPECT_NEAR(bound.radius, 2.25, kTolerance);
}

TEST(Bodies, RotatedBoxUsesPerAxisScaleAndPadding)
{
  robot_self_filter::shapes::Box shape(2.0, 4.0, 6.0);
  robot_self_filter::bodies::Box body(&shape);
  body.setScale(2.0, 0.5, 1.0);
  body.setPadding(0.1, 0.2, 0.3);

  tf2::Quaternion rotation;
  rotation.setRPY(0.0, 0.0, M_PI_2);
  tf2::Transform pose(rotation, tf2::Vector3(1.0, -2.0, 0.5));
  body.setPose(pose);

  EXPECT_TRUE(body.containsPoint(tf2::Vector3(1.0, 0.0, 0.5)));
  EXPECT_FALSE(body.containsPoint(tf2::Vector3(1.0, 0.100001, 0.5)));
  EXPECT_TRUE(body.containsPoint(tf2::Vector3(2.19, -2.0, 0.5)));
  EXPECT_FALSE(body.containsPoint(tf2::Vector3(2.21, -2.0, 0.5)));

  std::vector<tf2::Vector3> hits;
  ASSERT_TRUE(body.intersectsRay(
    tf2::Vector3(1.0, -10.0, 0.5), tf2::Vector3(0.0, 1.0, 0.0), &hits, 1));
  ASSERT_EQ(hits.size(), 1U);
  EXPECT_NEAR(hits.front().y(), -4.1, kTolerance);
}

TEST(Bodies, OptimizedBoxIntersectionMatchesLegacySlabAlgorithm)
{
  robot_self_filter::shapes::Box shape(3.0, 1.5, 2.25);
  robot_self_filter::bodies::Box body(&shape);
  body.setScale(1.2, 0.8, 1.4);
  body.setPadding(0.08, 0.03, 0.12);
  tf2::Quaternion rotation;
  rotation.setRPY(0.37, -0.51, 1.13);
  body.setPose(tf2::Transform(rotation, tf2::Vector3(1.2, -2.4, 0.7)));

  std::uint32_t random_state = 0xc0ffee42U;
  for (int sample = 0; sample < 20000; ++sample)
  {
    const tf2::Vector3 origin(
      randomRange(random_state, -12.0, 12.0),
      randomRange(random_state, -12.0, 12.0),
      randomRange(random_state, -12.0, 12.0));
    tf2::Vector3 direction(
      randomRange(random_state, -1.0, 1.0),
      randomRange(random_state, -1.0, 1.0),
      randomRange(random_state, -1.0, 1.0));
    if (direction.length2() < 1e-12)
      direction = tf2::Vector3(1.0, 0.0, 0.0);
    direction.normalize();

    std::vector<tf2::Vector3> legacy_hits;
    std::vector<tf2::Vector3> optimized_hits;
    const bool legacy_result = legacyBoxIntersectsRay(
      body, origin, direction, &legacy_hits);
    const bool optimized_result = body.intersectsRay(
      origin, direction, &optimized_hits, 1);
    ASSERT_EQ(optimized_result, legacy_result) << "sample " << sample;
    if (legacy_result)
    {
      ASSERT_EQ(optimized_hits.size(), legacy_hits.size()) << "sample " << sample;
      EXPECT_NEAR(optimized_hits[0].x(), legacy_hits[0].x(), 1e-8);
      EXPECT_NEAR(optimized_hits[0].y(), legacy_hits[0].y(), 1e-8);
      EXPECT_NEAR(optimized_hits[0].z(), legacy_hits[0].z(), 1e-8);
    }
  }
}

TEST(Bodies, CylinderUsesRadialAndVerticalScaleAndPadding)
{
  robot_self_filter::shapes::Cylinder shape(1.0, 4.0);
  robot_self_filter::bodies::Cylinder body(&shape);
  body.setScale(1.5, 0.5);
  body.setPadding(0.2, 0.3);

  tf2::Transform pose;
  pose.setIdentity();
  pose.setOrigin(tf2::Vector3(-1.0, 0.5, 2.0));
  body.setPose(pose);

  EXPECT_TRUE(body.containsPoint(tf2::Vector3(0.69, 0.5, 2.0)));
  EXPECT_FALSE(body.containsPoint(tf2::Vector3(0.71, 0.5, 2.0)));
  EXPECT_TRUE(body.containsPoint(tf2::Vector3(-1.0, 0.5, 3.29)));
  EXPECT_FALSE(body.containsPoint(tf2::Vector3(-1.0, 0.5, 3.31)));

  std::vector<tf2::Vector3> hits;
  ASSERT_TRUE(body.intersectsRay(
    tf2::Vector3(-5.0, 0.5, 2.0), tf2::Vector3(1.0, 0.0, 0.0), &hits, 1));
  ASSERT_EQ(hits.size(), 1U);
  EXPECT_NEAR(hits.front().x(), -2.7, kTolerance);
}

TEST(Bodies, ConvexMeshContainsAndIntersectsCube)
{
  const std::vector<tf2::Vector3> vertices = {
    {-1.0, -1.0, -1.0}, {1.0, -1.0, -1.0}, {1.0, 1.0, -1.0}, {-1.0, 1.0, -1.0},
    {-1.0, -1.0, 1.0}, {1.0, -1.0, 1.0}, {1.0, 1.0, 1.0}, {-1.0, 1.0, 1.0}};
  const std::vector<unsigned int> triangles = {
    0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7,
    0, 1, 5, 0, 5, 4, 1, 2, 6, 1, 6, 5,
    2, 3, 7, 2, 7, 6, 3, 0, 4, 3, 4, 7};

  std::unique_ptr<robot_self_filter::shapes::Mesh> shape(
    robot_self_filter::shapes::createMeshFromVertices(vertices, triangles));
  ASSERT_NE(shape, nullptr);
  robot_self_filter::bodies::ConvexMesh body(shape.get());

  tf2::Transform pose;
  pose.setIdentity();
  pose.setOrigin(tf2::Vector3(2.0, 0.0, 0.0));
  body.setPose(pose);

  EXPECT_TRUE(body.containsPoint(tf2::Vector3(2.0, 0.0, 0.0)));
  EXPECT_FALSE(body.containsPoint(tf2::Vector3(3.1, 0.0, 0.0)));

  std::vector<tf2::Vector3> hits;
  EXPECT_TRUE(body.intersectsRay(
    tf2::Vector3(-2.0, 0.0, 0.0), tf2::Vector3(1.0, 0.0, 0.0), &hits, 1));
  ASSERT_EQ(hits.size(), 1U);
  EXPECT_NEAR(hits.front().x(), 1.0, kTolerance);
}

// Bullet builds the hull in single precision.
constexpr double kHullTolerance = 1e-6;

std::unique_ptr<robot_self_filter::shapes::Mesh> makeBoxMesh(double hx, double hy, double hz)
{
  const std::vector<tf2::Vector3> vertices = {
    {-hx, -hy, -hz}, {hx, -hy, -hz}, {hx, hy, -hz}, {-hx, hy, -hz},
    {-hx, -hy, hz}, {hx, -hy, hz}, {hx, hy, hz}, {-hx, hy, hz}};
  const std::vector<unsigned int> triangles = {
    0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7,
    0, 1, 5, 0, 5, 4, 1, 2, 6, 1, 6, 5,
    2, 3, 7, 2, 7, 6, 3, 0, 4, 3, 4, 7};
  return std::unique_ptr<robot_self_filter::shapes::Mesh>(
    robot_self_filter::shapes::createMeshFromVertices(vertices, triangles));
}

// What ConvexMesh::intersectsRay computed before it clipped the ray against the hull's
// half-spaces: the nearest hull triangle in front of the origin (Moller-Trumbore).
bool hullTrianglesIntersectRay(
  const robot_self_filter::bodies::ConvexMesh &body,
  const tf2::Vector3 &origin,
  const tf2::Vector3 &direction,
  double &t_first)
{
  const auto &vertices = body.getScaledVertices();
  const auto &triangles = body.getTriangles();
  t_first = std::numeric_limits<double>::infinity();
  for (size_t i = 0; i + 2 < triangles.size(); i += 3)
  {
    const tf2::Vector3 a = body.getPose() * vertices[triangles[i]];
    const tf2::Vector3 e1 = body.getPose() * vertices[triangles[i + 1]] - a;
    const tf2::Vector3 e2 = body.getPose() * vertices[triangles[i + 2]] - a;
    const tf2::Vector3 p = direction.cross(e2);
    const double det = e1.dot(p);
    if (std::fabs(det) < 1e-12)
      continue;
    const tf2::Vector3 s = origin - a;
    const double u = s.dot(p) / det;
    if (u < 0.0 || u > 1.0)
      continue;
    const tf2::Vector3 q = s.cross(e1);
    const double v = direction.dot(q) / det;
    if (v < 0.0 || u + v > 1.0)
      continue;
    const double t = e2.dot(q) / det;
    if (t > 0.0)
      t_first = std::min(t_first, t);
  }
  return std::isfinite(t_first);
}

TEST(Bodies, ConvexMeshPadsEveryFaceAlongItsNormal)
{
  // A long thin link. Padding the vertices radially from the centroid would pad its sides less
  // and less towards its ends; moving the faces pads them evenly.
  auto shape = makeBoxMesh(1.0, 0.1, 0.1);
  ASSERT_NE(shape, nullptr);
  robot_self_filter::bodies::ConvexMesh body(shape.get());
  body.setPadding(0.05);

  tf2::Quaternion rotation;
  rotation.setRPY(0.0, 0.0, M_PI_2);  // long axis along world y, sides at world x 0.9..1.1
  body.setPose(tf2::Transform(rotation, tf2::Vector3(1.0, 2.0, 0.0)));

  // Side face, right next to the end of the link.
  EXPECT_TRUE(body.containsPoint(tf2::Vector3(0.851, 2.9, 0.0)));
  EXPECT_FALSE(body.containsPoint(tf2::Vector3(0.849, 2.9, 0.0)));
  // End face.
  EXPECT_TRUE(body.containsPoint(tf2::Vector3(1.0, 3.049, 0.0)));
  EXPECT_FALSE(body.containsPoint(tf2::Vector3(1.0, 3.051, 0.0)));

  std::vector<tf2::Vector3> hits;
  ASSERT_TRUE(body.intersectsRay(
    tf2::Vector3(-5.0, 2.9, 0.0), tf2::Vector3(1.0, 0.0, 0.0), &hits, 1));
  ASSERT_EQ(hits.size(), 1U);
  EXPECT_NEAR(hits[0].x(), 0.85, kHullTolerance);

  // count 0 asks for every crossing: in through one padded side, out through the other.
  hits.clear();
  ASSERT_TRUE(body.intersectsRay(
    tf2::Vector3(-5.0, 2.9, 0.0), tf2::Vector3(1.0, 0.0, 0.0), &hits, 0));
  ASSERT_EQ(hits.size(), 2U);
  EXPECT_NEAR(hits[0].x(), 0.85, kHullTolerance);
  EXPECT_NEAR(hits[1].x(), 1.15, kHullTolerance);

  // From inside there is only the way out.
  hits.clear();
  ASSERT_TRUE(body.intersectsRay(
    tf2::Vector3(1.0, 2.9, 0.0), tf2::Vector3(1.0, 0.0, 0.0), &hits, 0));
  ASSERT_EQ(hits.size(), 1U);
  EXPECT_NEAR(hits[0].x(), 1.15, kHullTolerance);

  // Pointing away from it, the hull is behind the ray.
  EXPECT_FALSE(body.intersectsRay(
    tf2::Vector3(1.5, 2.9, 0.0), tf2::Vector3(1.0, 0.0, 0.0), nullptr, 1));

  robot_self_filter::bodies::BoundingSphere bound;
  body.computeBoundingSphere(bound);
  EXPECT_NEAR(bound.center.x(), 1.0, kHullTolerance);
  EXPECT_NEAR(bound.center.y(), 2.0, kHullTolerance);
  EXPECT_NEAR(bound.radius, std::sqrt(1.0 + 0.01 + 0.01) + 0.05, kHullTolerance);
}

TEST(Bodies, ConvexMeshScalesAboutItsCentroid)
{
  // A tetrahedron's centroid (0.5, 0.25, 0.25) is not its bounding box's centre (1, 0.5, 0.5), so
  // scaling it about the centroid moves the hull out of a box that only grows about its centre.
  const std::vector<tf2::Vector3> vertices = {
    {0.0, 0.0, 0.0}, {2.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
  std::unique_ptr<robot_self_filter::shapes::Mesh> shape(
    robot_self_filter::shapes::createMeshFromVertices(
      vertices, {0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3}));
  ASSERT_NE(shape, nullptr);
  robot_self_filter::bodies::ConvexMesh body(shape.get());
  body.setScale(2.0);
  tf2::Transform pose;
  pose.setIdentity();
  body.setPose(pose);

  // (1.8, 0.02, 0.02) and (0.05, 0.925, 0.025), both inside the original, scaled about the centroid.
  EXPECT_TRUE(body.containsPoint(tf2::Vector3(3.1, -0.21, -0.21)));
  EXPECT_TRUE(body.containsPoint(tf2::Vector3(-0.4, 1.6, -0.2)));
  // Past the scaled tip at (3.5, -0.25, -0.25).
  EXPECT_FALSE(body.containsPoint(tf2::Vector3(3.6, -0.2, -0.2)));
  // Inside the scaled bounding box, outside the slanted face.
  EXPECT_FALSE(body.containsPoint(tf2::Vector3(2.0, 1.0, 1.0)));

  std::vector<tf2::Vector3> hits;
  ASSERT_TRUE(body.intersectsRay(
    tf2::Vector3(3.1, -5.0, -0.21), tf2::Vector3(0.0, 1.0, 0.0), &hits, 1));
  ASSERT_EQ(hits.size(), 1U);
  EXPECT_NEAR(hits[0].y(), -0.25, kHullTolerance);  // through the y = 0 face, scaled to -0.25
}

TEST(Bodies, ConvexMeshRayClippingMatchesHullTrianglesWhenUnpadded)
{
  std::uint32_t random_state = 0x5eed1234U;
  std::vector<tf2::Vector3> vertices;  // read as a triangle soup, so a multiple of three
  while (vertices.size() < 201U)
  {
    const tf2::Vector3 v(
      randomRange(random_state, -1.0, 1.0),
      randomRange(random_state, -1.0, 1.0),
      randomRange(random_state, -1.0, 1.0));
    if (v.length2() <= 1.0)
      vertices.emplace_back(0.3 * v.x(), 0.1 * v.y(), 0.15 * v.z());
  }
  std::unique_ptr<robot_self_filter::shapes::Mesh> shape(
    robot_self_filter::shapes::createMeshFromVertices(vertices));
  ASSERT_NE(shape, nullptr);
  robot_self_filter::bodies::ConvexMesh body(shape.get());
  ASSERT_GT(body.getTriangles().size(), 3U * 20U);
  tf2::Quaternion rotation;
  rotation.setRPY(0.4, -0.3, 1.1);
  body.setPose(tf2::Transform(rotation, tf2::Vector3(0.7, -0.2, 1.3)));

  int hit_count = 0;
  for (int sample = 0; sample < 20000; ++sample)
  {
    const tf2::Vector3 origin(
      randomRange(random_state, 0.7 - 0.6, 0.7 + 0.6),
      randomRange(random_state, -0.2 - 0.6, -0.2 + 0.6),
      randomRange(random_state, 1.3 - 0.6, 1.3 + 0.6));
    // Aimed around the hull, so rays hit, miss and graze it in comparable numbers.
    const tf2::Vector3 target(
      randomRange(random_state, 0.7 - 0.35, 0.7 + 0.35),
      randomRange(random_state, -0.2 - 0.35, -0.2 + 0.35),
      randomRange(random_state, 1.3 - 0.35, 1.3 + 0.35));
    tf2::Vector3 direction = target - origin;
    if (direction.length2() < 1e-12)
      direction = tf2::Vector3(1.0, 0.0, 0.0);
    direction.normalize();

    double t_expected = 0.0;
    const bool expected = hullTrianglesIntersectRay(body, origin, direction, t_expected);
    std::vector<tf2::Vector3> hits;
    ASSERT_EQ(body.intersectsRay(origin, direction, &hits, 1), expected) << "sample " << sample;
    if (expected)
    {
      ++hit_count;
      ASSERT_EQ(hits.size(), 1U) << "sample " << sample;
      EXPECT_NEAR((hits[0] - origin).dot(direction), t_expected, kHullTolerance)
        << "sample " << sample;
    }
  }
  // Both outcomes are exercised.
  EXPECT_GT(hit_count, 2000);
  EXPECT_LT(hit_count, 18000);
}

TEST(Shapes, RejectsInvalidTriangleTopologyAndIndices)
{
  const std::vector<tf2::Vector3> vertices = {
    {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}};
  EXPECT_EQ(robot_self_filter::shapes::createMeshFromVertices(vertices, {0, 1}), nullptr);
  EXPECT_EQ(robot_self_filter::shapes::createMeshFromVertices(vertices, {0, 1, 3}), nullptr);
}

TEST(Shapes, ValidatesBinaryStlBufferBeforeReading)
{
  std::vector<char> truncated(83U, 0);
  EXPECT_EQ(robot_self_filter::shapes::createMeshFromBinaryStlData(
    truncated.data(), static_cast<unsigned int>(truncated.size())), nullptr);

  std::vector<char> binary_stl(84U + 50U, 0);
  const std::uint32_t triangle_count = 1U;
  std::memcpy(binary_stl.data() + 80U, &triangle_count, sizeof(triangle_count));
  const float vertices[9] = {
    0.0F, 0.0F, 0.0F,
    1.0F, 0.0F, 0.0F,
    0.0F, 1.0F, 0.0F};
  std::memcpy(binary_stl.data() + 96U, vertices, sizeof(vertices));
  std::unique_ptr<robot_self_filter::shapes::Mesh> mesh(
    robot_self_filter::shapes::createMeshFromBinaryStlData(
      binary_stl.data(), static_cast<unsigned int>(binary_stl.size())));
  ASSERT_NE(mesh, nullptr);
  EXPECT_EQ(mesh->vertexCount, 3U);
  EXPECT_EQ(mesh->triangleCount, 1U);
}

TEST(Bodies, MergeBoundingSpheresHandlesEmptyNestedAndSeparatedInputs)
{
  robot_self_filter::bodies::BoundingSphere merged;
  robot_self_filter::bodies::mergeBoundingSpheres({}, merged);
  EXPECT_DOUBLE_EQ(merged.radius, 0.0);
  EXPECT_EQ(merged.center, tf2::Vector3(0.0, 0.0, 0.0));

  const std::vector<robot_self_filter::bodies::BoundingSphere> spheres = {
    {tf2::Vector3(0.0, 0.0, 0.0), 2.0},
    {tf2::Vector3(0.5, 0.0, 0.0), 0.25},
    {tf2::Vector3(6.0, 0.0, 0.0), 2.0}};
  robot_self_filter::bodies::mergeBoundingSpheres(spheres, merged);
  EXPECT_NEAR(merged.center.x(), 3.0, kTolerance);
  EXPECT_NEAR(merged.radius, 5.0, kTolerance);
}

}  // namespace
