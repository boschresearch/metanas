"""Module for data interface definitions
Copyright (c) 2021 Robert Bosch GmbH

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

"""

from abc import ABC
from collections import namedtuple


Task = namedtuple("Task", ["train_loader", "valid_loader", "test_loader"])


class TaskDistribution(ABC):
    """Base class to sample tasks for meta training"""

    def sample_meta_train(self):
        """Sample a meta batch for training

        Returns:
            A list of tasks
        """
        raise NotImplementedError

    def sample_meta_valid(self):
        """Sample a meta batch for validation

        Returns:
            A list of tasks
        """
        raise NotImplementedError

    def sample_meta_test(self):
        """Sample a meta batch for testing

        Returns:
            A list of tasks
        """
        raise NotImplementedError
