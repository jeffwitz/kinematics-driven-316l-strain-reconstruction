function Pandoc(doc)
  if #doc.blocks > 0 and doc.blocks[1].t == "Header" and doc.blocks[1].level == 1 then
    table.remove(doc.blocks, 1)
  end
  return doc
end
