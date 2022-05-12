import assert from 'node:assert/strict'
import { fromMarkdown } from 'mdast-util-from-markdown'
import { gfmTaskListItem } from 'micromark-extension-gfm-task-list-item'
import { gfmTaskListItemFromMarkdown } from 'mdast-util-gfm-task-list-item'

const findFirstValue = (object) => {
  if (object.value != null) {
    return object.value
  }
  if (object.children) {
    for (const item of object.children) {
      const res = findFirstValue(item)
      if (res != null) {
        return res
      }
    }
  }
}

const parsePullRequstBody = (doc, hintHeading) => {
  const tree = fromMarkdown(doc, {
    extensions: [gfmTaskListItem],
    mdastExtensions: [gfmTaskListItemFromMarkdown]
  })
  assert.equal(tree.type, 'root')

  const heading = fromMarkdown(hintHeading)
  assert.equal(heading.children[0].type, 'heading')
  const headingDepth = heading.children[0].depth
  const headingValue = findFirstValue(heading.children[0])

  const targetIdx = tree.children.findIndex(x => {
    if (x.type === 'heading' && x.depth === headingDepth) {
      return findFirstValue(x) === headingValue
    }
    return false
  })
  assert.notEqual(targetIdx, -1)

  const listNode = tree.children[targetIdx + 1]
  assert.equal(listNode.type, 'list')

  const selectedItems = listNode.children.filter(x => x.checked)
  return selectedItems.map((x, idx) => [findFirstValue(x), idx])
}

export { parsePullRequstBody }
