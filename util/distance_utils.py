import time
import torch

def mapping_base_on_hypersphere_constraint(x_prev, step_size, norm_gradient, x_t_mean, r, device='cuda:0'):
    hyperplane_point = x_prev - step_size * norm_gradient
    hyperplane_bias = -(norm_gradient * hyperplane_point).sum()
    # print(f'hyperplane_bias1: {hyperplane_bias.item()}')
    # hyperplane_point = hyperplane_point.reshape(-1)
    hyperplane_normal = norm_gradient.reshape(-1)
    # hyperplane_bias = torch.matmul(hyperplane_normal, hyperplane_point)
    # print(f'hyperplane_bias2: {hyperplane_bias.item()}')
    hypersphere_radius = r
    hypersphere_center = x_t_mean.reshape(-1)
    intersection_point = find_intersection(hyperplane_normal, hyperplane_bias, hypersphere_center, hypersphere_radius, device=device)
    return intersection_point


def projection_into_hyperplane(hyperplane_normal, hyperplane_bias, point):
    distance = (torch.matmul(hyperplane_normal, point) + hyperplane_bias) / hyperplane_normal.pow(2).sum()
    # print(f'hyperplane_normal.pow(2).sum():{hyperplane_normal.pow(2).sum()}')
    projection_point = point - distance * hyperplane_normal
    return distance, projection_point

# Function to find the intersection points
def find_intersection(hyperplane_normal, hyperplane_bias, hypersphere_center, hypersphere_radius, exp=1e-10, device='cuda:0'):
    # Calculate the distance between the hyperplane and the center of the hypersphere
    # start = time.time()
    distance, projection_point = projection_into_hyperplane(hyperplane_normal, hyperplane_bias, hypersphere_center)
    # distance_time1 = time.time()
    # print(f'distance_time1:{distance_time1-start}')

    # Check if the hypersphere intersects with the hyperplane
    if torch.abs(distance) < hypersphere_radius:
        # Calculate the intersection points
        # intersection_points = hypersphere_center + hyperplane_normal * (hypersphere_radius - distance)
        sample_point = torch.normal(mean=projection_point).to(device)
        _, sample_point_projected = projection_into_hyperplane(hyperplane_normal, hyperplane_bias, sample_point)
        # distance_time2 = time.time()
        # print(f'distance_time2:{distance_time2 - distance_time1}')
        scaled_direction = (sample_point_projected - projection_point) / (torch.norm(sample_point_projected - projection_point) + exp)
        intersection_point = projection_point + (hypersphere_radius.pow(2) - distance.pow(2)).sqrt() * scaled_direction
        # end = time.time()
        # print(f'distance_time2 to end:{end - distance_time2}')
        return intersection_point
    else:
        # print(f'distance: {distance} hypersphere_radius: {hypersphere_radius}')
        return None



if __name__ == '__main__':

    # Find the intersection points
    device = 'cuda:0'

    # Set the dimensionality of the space
    dimension = 256 * 256

    # Generate a random normal vector for the hyperplane
    hyperplane_normal = torch.randn(dimension).to(device)

    # Normalize the normal vector
    # hyperplane_normal /= torch.norm(hyperplane_normal).to(device)

    # Generate a random point for the hyperplane
    hyperplane_bias = torch.randn(1).to(device)

    # Generate a random center for the hypersphere
    hypersphere_center = torch.randn(dimension).to(device)

    # Generate a random radius for the hypersphere
    # hypersphere_radius = torch.rand(1)
    hypersphere_radius = (torch.ones(1) * 100).to(device)

    # print(f'hypersphere_radius device:{hypersphere_radius.device}')
    for i in range(10):
        start = time.time()
        intersection_points = find_intersection(hyperplane_normal, hyperplane_bias, hypersphere_center, hypersphere_radius)
        end = time.time()
        print(f'total_time{i}:{end-start}')

        if intersection_points is not None:
        # Print the results
            print(f"iteration i:{i}\n\n")
            print("Hyperplane Normal:", hyperplane_normal)
            print("Hyperplane Bias:", hyperplane_bias)
            print("Hypersphere Center:", hypersphere_center)
            print("Hypersphere Radius:", hypersphere_radius)
            print("Intersection Points:", intersection_points)

            print("Intersection Points distance to center:", torch.norm(intersection_points-hypersphere_center))
            hyperplane_distance = (torch.matmul(hyperplane_normal, intersection_points) + hyperplane_bias) / torch.norm(hyperplane_normal)
            print("Intersection Points distance to hyperplane:", hyperplane_distance)


    # Testing point-hyperplane distance
    # distance, projection_point = projection_into_hyperplane(hyperplane_normal, hyperplane_bias, hypersphere_center)
    # hyperplane_distance = (torch.matmul(hyperplane_normal, projection_point) + hyperplane_bias) / torch.norm(hyperplane_normal)
    # print(f'distance:{distance}')
    # print(f'projection_point hyperplane distance:{hyperplane_distance}')


