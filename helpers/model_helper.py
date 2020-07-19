import os

def find_snapshot(snapshot_dir, snapshot_num=None):
  found_snapshot = False
  if os.path.exists(snapshot_dir):
    print('Found snapshot directory!')
    snapshots = sorted(os.listdir(snapshot_dir))
    if snapshot_num is not None:
      for snapshot in snapshots:
        if str(snapshot_num) in snapshot:
          model_name = snapshot
          found_snapshot = True
          break
    else:
      model_name = snapshots[-1]
      found_snapshot = True
    if found_snapshot:
      print('Found snapshot: Loading snapshot ' + model_name + '...')
    else:
      print('Snapshot number does not exist! Please choose from:')
      print(snapshots)
      exit()
  else:
    print('Did not find snapshot directory! Training from scratch...')
    model_name = None

  return model_name