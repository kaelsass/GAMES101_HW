#include <algorithm>
#include <cassert>
#include <map>
#include "BVH.hpp"

BVHAccel::BVHAccel(std::vector<Object*> p, int maxPrimsInNode,
                   SplitMethod splitMethod)
    : maxPrimsInNode(std::min(255, maxPrimsInNode)), splitMethod(splitMethod),
      primitives(std::move(p))
{
    time_t start, stop;
    time(&start);
    if (primitives.empty())
        return;

    root = recursiveBuild(primitives);

    time(&stop);
    double diff = difftime(stop, start);
    int hrs = (int)diff / 3600;
    int mins = ((int)diff / 60) - (hrs * 60);
    int secs = (int)diff - (hrs * 3600) - (mins * 60);

    printf(
        "\rBVH Generation complete: \nTime Taken: %i hrs, %i mins, %i secs\n\n",
        hrs, mins, secs);
}

BVHBuildNode* BVHAccel::recursiveBuild(std::vector<Object*> objects)
{
    BVHBuildNode* node = new BVHBuildNode();

    // Compute bounds of all primitives in BVH node
    Bounds3 bounds;
    for (int i = 0; i < objects.size(); ++i)
        bounds = Union(bounds, objects[i]->getBounds());
    if (objects.size() == 1) {
        // Create leaf _BVHBuildNode_
        node->bounds = objects[0]->getBounds();
        node->object = objects[0];
        node->left = nullptr;
        node->right = nullptr;
        return node;
    }
    else if (objects.size() == 2) {
        node->left = recursiveBuild(std::vector{objects[0]});
        node->right = recursiveBuild(std::vector{objects[1]});

        node->bounds = Union(node->left->bounds, node->right->bounds);
        return node;
    }
    else {
        std::vector<Object*> leftshapes;
        std::vector<Object*> rightshapes;
        Bounds3 centroidBounds;
        for (int i = 0; i < objects.size(); ++i)
            centroidBounds =
                    Union(centroidBounds, objects[i]->getBounds().Centroid());
        switch(splitMethod) {
            case SplitMethod::NAIVE: {
                int dim = centroidBounds.maxExtent();
                switch (dim) {
                    case 0:
                        std::sort(objects.begin(), objects.end(), [](auto f1, auto f2) {
                            return f1->getBounds().Centroid().x <
                                   f2->getBounds().Centroid().x;
                        });
                        break;
                    case 1:
                        std::sort(objects.begin(), objects.end(), [](auto f1, auto f2) {
                            return f1->getBounds().Centroid().y <
                                   f2->getBounds().Centroid().y;
                        });
                        break;
                    case 2:
                        std::sort(objects.begin(), objects.end(), [](auto f1, auto f2) {
                            return f1->getBounds().Centroid().z <
                                   f2->getBounds().Centroid().z;
                        });
                        break;
                }

                auto beginning = objects.begin();
                auto middling = objects.begin() + (objects.size() / 2);
                auto ending = objects.end();

                leftshapes = std::vector<Object*>(beginning, middling);
                rightshapes = std::vector<Object*>(middling, ending);
            }
            break;
            case SplitMethod::SAH: {
                int bucketCount = 12;
                int minAxis = 0;
                int minIndex = 0;
                float minCost = std::numeric_limits<float>::infinity();
                std::map<int, std::map<int, int>> indexMap;
                for (int axis = 0; axis < 3; axis++) {
                    std::vector<Bounds3> boundsBucket(bucketCount);
                    std::vector<int> countBucket(bucketCount);
                    for (int i = 0; i < objects.size(); i++) {
                        int bucketIndex = bucketCount * centroidBounds.Offset(objects[i]->getBounds().Centroid())[axis];
                        bucketIndex = std::clamp(bucketIndex, 0, bucketCount - 1);
                        Bounds3 curBounds = boundsBucket[bucketIndex];
                        curBounds = Union(curBounds, objects[i]->getBounds());
                        boundsBucket[bucketIndex] = curBounds;
                        countBucket[bucketIndex]++;
                        indexMap[axis][i] = bucketIndex;
                    }
                    std::vector<Bounds3> leftBounds(boundsBucket), rightBounds(boundsBucket);
                    for (int partition = 1; partition < bucketCount; partition++) {
                        leftBounds[partition] = Union(leftBounds[partition - 1], leftBounds[partition]);
                    }
                    for (int partition = bucketCount - 2; partition >= 0; partition--) {
                        rightBounds[partition] = Union(rightBounds[partition + 1], rightBounds[partition]);
                    }
                    int leftBucketCount = 0;
                    int rightBucketCount = objects.size();
                    for (int partition = 0; partition < bucketCount - 1; partition++) {
                        leftBucketCount += countBucket[partition];
                        rightBucketCount -= countBucket[partition];
                        float cost = (leftBucketCount * leftBounds[partition].SurfaceArea() +
                                       rightBucketCount * rightBounds[partition + 1].SurfaceArea()) /
                                      bounds.SurfaceArea();
                        if (minCost > cost) {
                            minCost = cost;
                            minAxis = axis;
                            minIndex = partition;
                        }
                    }
                }
                for (int i = 0; i < objects.size(); i++) {
                    if (indexMap[minAxis][i] <= minIndex) {
                        leftshapes.push_back(objects[i]);
                    } else {
                        rightshapes.push_back(objects[i]);
                    }
                }
            }
            break;
        }

        assert(objects.size() == (leftshapes.size() + rightshapes.size()));

        node->left = recursiveBuild(leftshapes);
        node->right = recursiveBuild(rightshapes);

        node->bounds = Union(node->left->bounds, node->right->bounds);
    }

    return node;
}

Intersection BVHAccel::Intersect(const Ray& ray) const
{
    Intersection isect;
    if (!root)
        return isect;
    isect = BVHAccel::getIntersection(root, ray);
    return isect;
}

Intersection BVHAccel::getIntersection(BVHBuildNode* node, const Ray& ray) const
{
    // TODO Traverse the BVH to find intersection
    Intersection isect;
    isect.happened = false;
    if (node == nullptr) {
        return isect;
    }
    if (!node->bounds.IntersectP(ray, ray.direction_inv,
                                 {ray.direction.x > 0, ray.direction.y > 0, ray.direction.z > 0})) {
        return isect;
    }
    if (node->left == nullptr && node->right == nullptr) {
        return node->object->getIntersection(ray);
    }
    Intersection hitLeft = getIntersection(node->left, ray);
    Intersection hitRight = getIntersection(node->right, ray);
    return hitLeft.distance < hitRight.distance ? hitLeft : hitRight;
}