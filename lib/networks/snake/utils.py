import torch
import math
# from detectron2.structures import Boxes, Instances

# def add_ground_truth_to_pred_boxes(gt_boxes, proposals):
#     """
#     Call `add_ground_truth_to_proposals_single_image` for all images.

#     Args:
#         gt_boxes(list[Boxes]): list of N elements. Element i is a Boxes
#             representing the gound-truth for image i.
#         proposals (list[Instances]): list of N elements. Element i is a Instances
#             representing the proposals for image i.

#     Returns:
#         list[Instances]: list of N Instances. Each is the proposals for the image,
#             with field "proposal_boxes" and "objectness_logits".
#     """
#     assert gt_boxes is not None

#     assert len(proposals) == len(gt_boxes)
#     if len(proposals) == 0:
#         return proposals

#     return [
#         add_ground_truth_to_pred_boxes_single_image(gt_boxes_i, proposals_i)
#         for gt_boxes_i, proposals_i in zip(gt_boxes, proposals)
#     ]


# def add_ground_truth_to_pred_boxes_single_image(gt_boxes, proposals):
#     """
#     Augment `proposals` with ground-truth boxes from `gt_boxes`.

#     Args:
#         Same as `add_ground_truth_to_proposals`, but with gt_boxes and proposals
#         per image.

#     Returns:
#         Same as `add_ground_truth_to_proposals`, but for only one image.
#     """
#     device = proposals.pred_boxes.device
#     # Assign all ground-truth boxes an objectness logit corresponding to
#     # P(object) = sigmoid(logit) =~ 1.
#     gt_logit_value = math.log((1.0 - 1e-10) / (1 - (1.0 - 1e-10)))
#     gt_logits = gt_logit_value * torch.ones(len(gt_boxes), device=device)

#     # Concatenating gt_boxes with proposals requires them to have the same fields
#     gt_proposal = Instances(proposals.image_size)
#     gt_proposal.pred_boxes = gt_boxes
#     gt_proposal.scores = gt_logits
#     new_proposals = Instances.cat([proposals, gt_proposal])

#     return new_proposals

def patch2masks(patch, scale, patch_size, mask_size):
    """
    assemble patches to obtain an entire mask
    Args:
        mask_rc: A tensor of shape [B*num_patch,patch_size,patch_size]
        scale: A NxN mask size is divided into scale x scale patches
        patch_size: size of each patch
        mask_size: size of masks generated by PatchDCT

    Returns:
        A tensor of shape [B,mask_size,mask_size].The masks obtain assemble by patches
    """
    patch = patch.reshape(-1, scale, scale, patch_size, patch_size)
    patch = patch.permute(0, 1, 2, 4, 3)
    patch = patch.reshape(-1, scale, mask_size, patch_size)
    patch = patch.permute(0, 1, 3, 2)
    mask = patch.reshape(-1, mask_size, mask_size)
    return mask

def masks2patch(masks_per_image, scale, patch_size, mask_size):
    """

    Args:
        masks_per_image: A tensor of shape [B,mask_size,mask_size]
        scale: A NxN mask size is divided into scale x scale patches
        patch_size: size of each patch
        mask_size: size of masks generated by PatchDCT

    Returns:
        patches_per_image: A tensor of shape [B*num_patch,patch_size,patch_size]. The patches obtained by masks

    """
    masks_per_image = masks_per_image.reshape(-1, scale, patch_size,mask_size)
    masks_per_image = masks_per_image.permute(0, 1, 3, 2)
    masks_per_image = masks_per_image.reshape(-1, scale, scale, patch_size, patch_size)
    masks_per_image = masks_per_image.permute(0, 1, 2, 4, 3)
    patches_per_image = masks_per_image.reshape(-1,patch_size, patch_size)
    return patches_per_image
