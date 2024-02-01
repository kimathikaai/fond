import math
from typing import List


def create_domains_1(
    num_classes: int, num_linked: int, num_domains: int
) -> List[List[int]]:
    """
    Args:
        num_classes: number of overall classes within all the domains
        num_linked: number of classes that are linked to an individual domain
        num_domains: number of different domains, including test domain
    """
    assert num_linked < num_classes
    domain_shared = [i for i in range(num_linked, num_classes)]
    print(f"Shared classes: {domain_shared}")
    num_train_domains = num_domains - 1
    domains = [domain_shared.copy() for i in range(num_train_domains)]

    for class_idx in range(num_linked):
        domain_idx = class_idx % num_train_domains
        domains[domain_idx].append(class_idx)
    return domains


def create_domains_2(
    num_classes: int, num_linked_ratio: int, num_domains: int
) -> List[List[int]]:
    """
    Args:
        num_classes: number of overall classes within all the domains
        num_linked_ratio: ratio of linked classes to the total number of classes
        num_domains: number of different domains, including test domain
    """
    num_linked = math.floor(num_linked_ratio * num_classes)
    return create_domains_1(num_classes, num_linked, num_domains)


if __name__ == "__main__":
    # Testing same classes with different overlap
    domains = create_domains_1(num_classes=10, num_linked=3, num_domains=4)
    print(domains)
    domains = create_domains_1(num_classes=10, num_linked=5, num_domains=4)
    print(domains)

    # Testing different classes with same overlap percentage
    domains = create_domains_2(num_classes=10, num_linked_ratio=0.2, num_domains=4)
    print(domains)
    domains = create_domains_2(num_classes=5, num_linked_ratio=0.2, num_domains=4)
    print(domains)
